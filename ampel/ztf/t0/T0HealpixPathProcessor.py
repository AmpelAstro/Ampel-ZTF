#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/t0/T0HealpixPathProcessor.py
# License           : BSD-3-Clause
# Author            : mf <mf@physik.hu-berlin.de>
# Date              : 13.10.2021
# Last Modified Date: 27.04.2022
# Last Modified By  : jn <jno@physik.hu-berlin.de>

import tempfile, collections, requests
import numpy as np
import healpy as hp
from datetime import datetime
from signal import SIGINT, SIGTERM, signal
from typing import List, Tuple, Union

from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.alert.BaseAlertSupplier import BaseAlertSupplier
from ampel.alert.AlertConsumer import AlertConsumer
from ampel.alert.AlertConsumerError import AlertConsumerError
from ampel.alert.AlertConsumerMetrics import stat_accepted, stat_alerts, stat_time
from ampel.core.EventHandler import EventHandler
from ampel.ingest.ChainedIngestionHandler import ChainedIngestionHandler
from ampel.log import VERBOSE, LogFlag
from ampel.log.AmpelLogger import AmpelLogger
from ampel.log.AmpelLoggingError import AmpelLoggingError
from ampel.log.LightLogRecord import LightLogRecord
from ampel.log.utils import report_exception
from ampel.model.ingest.CompilerOptions import CompilerOptions
from ampel.model.UnitModel import UnitModel
from ampel.mongo.update.DBUpdatesBuffer import DBUpdatesBuffer
from pymongo.errors import PyMongoError


class T0HealpixPathProcessor(AlertConsumer):
    """
    Download a Healpix map from a provided URL and process.

    """

    compiler_opts: CompilerOptions

    # Process pixels with p-values lower than this limit
    pvalue_limit: float = 0.9

    # Name (signifier)
    map_name: str

    # URL for healpix retrieval
    map_url: str
    scratch_dir: str    # Local dir where map is saved. File with this name del

    # Max size of each api query
    querysize: int = 1000


    def __init__(self, **kwargs) -> None:
        if not kwargs.get("compiler_opts"):
            kwargs["compiler_opts"] = CompilerOptions()
        super().__init__(**kwargs)


    def proceed(self, event_hdlr: EventHandler) -> int:
        stats = {
            "alerts": stat_alerts,
            "accepted": stat_accepted.labels("any"),
        }
        run_id = self.context.new_run_id()
        logger = AmpelLogger.from_profile(
            self.context,
            self.log_profile,
            run_id,
            base_flag=LogFlag.T0 | LogFlag.CORE | self.base_log_flag,
            force_refresh=True,
        )

        if logger.verbose:
            logger.log(VERBOSE, "Pre-run setup")

        # DBLoggingHandler formats, saves and pushes log records into the DB
        if db_logging_handler := logger.get_db_logging_handler():
            db_logging_handler.auto_flush = False

        updates_buffer = DBUpdatesBuffer(
            self._ampel_db,
            run_id,
            logger,
            error_callback=self.set_cancel_run,
            catch_signals=False,  # we do it ourself
            max_size=self.updates_buffer_size,
        )

        any_filter = any([fb.filter_model for fb in self._fbh.filter_blocks])
        # if bypassing filters, track passing rates at top level
        if not any_filter:
            stats["filter_accepted"] = [
                stat_accepted.labels(channel)
                for channel in self._fbh.chan_names
            ]

        # Retrieve mapfile.
        temp = tempfile.NamedTemporaryFile(prefix="t0Healpix_", dir=self.scratch_dir, delete=False)
        logger.info('Downloading map', extra={'url': self.map_url, 'tmpfile': temp.name})
        map_data = requests.get(self.map_url)
        with open(temp.name, 'wb') as fh:
            fh.write(map_data.content)

        # Process map
        hpx, headers = hp.read_map(temp.name, h=True, nest=True)
        trigger_time = [datetime.fromisoformat(header[1]) for header in headers if header[0] == 'DATE-OBS'][0]
        # nside = int( ah.npix_to_nside(len(hpx)) )
        nside = int(hp.npix2nside(len(hpx)))

        # Find credible levels
        idx = np.flipud(np.argsort(hpx))
        sorted_credible_levels = np.cumsum(hpx[idx].astype(float))
        credible_levels = np.empty_like(sorted_credible_levels)
        credible_levels[idx] = sorted_credible_levels
        healpix_pvalues = list(credible_levels)

        # Create mask for pixel selection
        mask = np.zeros(len(credible_levels), int)
        mask[credible_levels <= self.pvalue_limit] = 1
        ipix = mask.nonzero()[0].tolist()
        query_list = [ipix[i:i+self.querysize] for i in range(0, len(ipix), self.querysize)]

        logger.info('healpix preprocces',extra={'trigger_time':trigger_time,'nside':nside, 'queries':len(query_list)})

        # Unit directives might use
#        print(self.directives)
        def update_value(d, old_val, new_val):
            for k, v in d.items():
                if v==old_val:
                    d[k]=new_val
                elif isinstance(v, collections.Mapping):
                    d[k] = update_value(v, old_val, new_val)
                elif isinstance(v, list):  # also need to account for tuple?
                    d[k] = [update_value[l, old_val, new_val] for l in v]
                else:
                    d[k] = v
            return d
#        foo = update_value(self.directives, 'HealpixTriggerTime', 'CHANGED')
#        print(foo)
#        sys.exit('did it work?')


        # Setup ingesters
        ing_hdlr = ChainedIngestionHandler(
            self.context, self.shaper, self.directives, updates_buffer,
            run_id, tier = 0, logger = logger, database = self.database,
            trace_id = {'alertconsumer': self._trace_id},
            compiler_opts = self.compiler_opts or CompilerOptions()
        )


        iter_max = self.iter_max
        if self.iter_max != self.__class__.iter_max:
            logger.info(f"Using custom iter_max: {self.iter_max}")

        self._cancel_run = 0
        iter_count = 0
        err = 0

        assert self._fbh.chan_names is not None
        reduced_chan_names: Union[str, List[str]] = (
            self._fbh.chan_names[0]
            if len(self._fbh.chan_names) == 1
            else self._fbh.chan_names
        )
        fblocks = self._fbh.filter_blocks

        if any_filter:
            filter_results: List[Tuple[int, Union[bool, int]]] = []
        else:
            filter_results = [(i, True) for i, fb in enumerate(fblocks)]

        # Builds set of stock ids for autocomplete, if needed
        self._fbh.ready(logger, run_id)

        logger.log(self.shout, "Processing alerts", extra={"r": run_id})

        try:
            updates_buffer.start()
            chatty_interrupt = self.chatty_interrupt
            register_signal = self.register_signal

            # Iterate over queries
            for i, pixel_query in enumerate(query_list):
                logger.debug( f"query {i} of {len(query_list)}" )
                # Update alert supplier
                # NB: since BaseAlertSupplier uses AuxUnitRegister to instantiate its loader, it doesn't
                # have access to the secrets store. Work around this by explicitly instantiating and
                # setting a loader through the context.
                self.alert_supplier = self.context.loader.new(self.supplier, unit_type=BaseAlertSupplier)
                assert isinstance(supplier_config := self.supplier.config, dict)
                loader = self.context.loader.new(
                    UnitModel(**supplier_config["loader"]),
                    unit_type=AbsAlertLoader, # type: ignore[type-var]
                    stream={"nside": nside, "pixels": pixel_query, "time": trigger_time}
                )
                self.alert_supplier.alert_loader = loader

                # Iterate over alerts from query
                for alert in self.alert_supplier:

                    # Allow execution to complete for this alert (loop exited after ingestion of current alert)
                    signal(SIGINT, register_signal)
                    signal(SIGTERM, register_signal)

                    # Associate upcoming log entries with the current transient id
                    stock_id = alert.stock

                    if any_filter:

                        filter_results = []

                        # Loop through filter blocks
                        for fblock in fblocks:
                            try:
                                # Apply filter (returns None/False in case of rejection or True/int in case of match)
                                res = fblock.filter(alert)
                                if res[1]:
                                    filter_results.append(res)  # type: ignore[arg-type]

                            # Unrecoverable (logging related) errors
                            except (PyMongoError, AmpelLoggingError) as e:
                                print(
                                    "%s: abording run() procedure"
                                    % e.__class__.__name__
                                )
                                self._report_ap_error(e, event_hdlr, logger, extra={"a": alert.id})
                                raise e

                            # Possibly tolerable errors (could be an error from a contributed filter)
                            except Exception as e:

                                if db_logging_handler:
                                    fblock.forward(
                                        db_logging_handler,
                                        stock=stock_id,
                                        extra={"a": alert.id},
                                    )
                                self._report_ap_error(
                                    e, event_hdlr, logger, extra={
                                        "a": alert.id,
                                        "section": "filter",
                                        "c": fblock.channel,
                                    }
                                )

                                if self.raise_exc:
                                    raise e
                                else:
                                    if self.error_max:
                                        err += 1
                                    if err == self.error_max:
                                        logger.error(
                                            "Max number of error reached, breaking alert processing"
                                        )
                                        self.set_cancel_run(
                                            AlertConsumerError.TOO_MANY_ERRORS
                                        )
                    else:
                        # if bypassing filters, track passing rates at top level
                        for counter in stats["filter_accepted"]:
                            counter.inc()

                    if filter_results:

                        stats["accepted"].inc()

                        # Determine p-value for being associated with Healpix map
                        # Should this be located here, in filter or loader?
                        # Also - should handle missing coord valus gracefully
                        if len( pos:= alert.get_tuples('ra', 'dec', filters=
                            [{'attribute': 'magpsf', 'operator': 'is not', 'value': None}]) )>0:
                            ra = np.mean( [p[0] for p in pos] )
                            dec = np.mean( [p[1] for p in pos] )
                            theta = 0.5 * np.pi - np.deg2rad(dec)
                            phi = np.deg2rad(ra)
                            alertpix = hp.pixelfunc.ang2pix(hp.npix2nside(len(healpix_pvalues)), theta, phi, nest=True)
                            alert_pvalue = healpix_pvalues[alertpix]
                        else:
                            # Should this be possible - throw error?
                            alert_pvalue = None

                        try:
                            with stat_time.labels("ingest").time():
                                ing_hdlr.ingest(
                                    alert.datapoints,
                                    filter_results,
                                    stock_id,
                                    alert.tag,
                                    {"alert": alert.id, "healpix": {
                                        'map': self.map_name,
                                        'pvalue': alert_pvalue,
                                        'nside': nside,
                                        'time': trigger_time.isoformat(),
                                        }
                                    },
                                )
                        except (PyMongoError, AmpelLoggingError) as e:
                            print(
                                "%s: abording run() procedure"
                                % e.__class__.__name__
                            )
                            self._report_ap_error(e, event_hdlr, logger, extra={"a": alert.id})
                            raise e

                        except Exception as e:

                            self._report_ap_error(
                                e, event_hdlr, logger, extra={
                                    "a": alert.id, "section": "ingest",
                                    "c": [self.directives[el[0]].channel for el in filter_results]
                                }
                            )

                            if self.raise_exc:
                                raise e

                            if self.error_max:
                                err += 1

                            if err == self.error_max:
                                logger.error(
                                    "Max number of error reached, breaking alert processing"
                                )
                                self.set_cancel_run(
                                    AlertConsumerError.TOO_MANY_ERRORS
                                    )

                    else:

                        # All channels reject this alert
                        # no log entries goes into the main logs collection sinces those are redirected to Ampel_rej.

                        # So we add a notification manually. For that, we don't use logger
                        # cause rejection messages were alreary logged into the console
                        # by the StreamHandler in channel specific RecordBufferingHandler instances.
                        # So we address directly db_logging_handler, and for that, we create
                        # a LogDocument manually.
                        lr = LightLogRecord(
                            logger.name, LogFlag.INFO | logger.base_flag
                        )
                        lr.stock = stock_id
                        lr.channel = reduced_chan_names  # type: ignore[assignment]
                        lr.extra = {"a": alert.id, "allout": True}
                        if db_logging_handler:
                            db_logging_handler.handle(lr)

                    iter_count += 1
                    stats["alerts"].inc()

                    updates_buffer.check_push()
                    if db_logging_handler:
                        db_logging_handler.check_flush()

                    if iter_count == iter_max:
                        logger.info("Reached max number of iterations")
                        break

                    # Exit if so requested (SIGINT, error registered by DBUpdatesBuffer, ...)
                    if self._cancel_run > 0:
                        break

                    # Restore system default sig handling so that KeyBoardInterrupt
                    # can be raised during supplier execution
                    signal(SIGINT, chatty_interrupt)
                    signal(SIGTERM, chatty_interrupt)

        # Executed if SIGINT was sent during supplier execution
        except KeyboardInterrupt:
            pass

        except Exception as e:
            # Try to insert doc into trouble collection (raises no exception)
            # Possible exception will be logged out to console in any case
            event_hdlr.add_extra(overwrite=True, success=False)
            report_exception(self._ampel_db, logger, exc=e)

        # Also executed after SIGINT and SIGTERM
        finally:

            updates_buffer.stop()

            if self._cancel_run > 0:
                print("")
                logger.info("Processing interrupted")
            else:
                logger.log(self.shout, "Processing completed")

            try:

                # Flush loggers
                logger.flush()

                # Flush registers and rejected log handlers
                self._fbh.done()

                event_hdlr.update()

            except Exception as e:

                # Try to insert doc into trouble collection (raises no exception)
                # Possible exception will be logged out to console in any case
                report_exception(self._ampel_db, logger, exc=e)

        # Return number of processed alerts
        return iter_count

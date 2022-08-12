#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/ampel/ztf/t0/T0HealpixProcessor.py
# License           : BSD-3-Clause
# Author            : mf <mf@physik.hu-berlin.de>
# Date              : 13.10.2021
# Last Modified Date: 27.04.2021
# Last Modified By  : jnordin <jnordin@physik.hu-berlin.de>

import numpy as np
import healpy as hp
from signal import SIGINT, SIGTERM, signal
from typing import Any, Dict, List, Optional, Tuple, Union

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
from ampel.mongo.update.DBUpdatesBuffer import DBUpdatesBuffer
from pymongo.errors import PyMongoError


class T0HealpixProcessor(AlertConsumer):
    """ """

#    healpix: List[int]
#    nside: int
#    time: datetime
#    with_history: bool = False
    healpix_log: Dict[str, Any]
    healpix_pvalues: List[float]
    compiler_opts: CompilerOptions

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
        # v1
#        self.alert_supplier.set_logger(logger)
#        self.alert_supplier.set_time(self.time)
#        self.alert_supplier.set_healpix(self.healpix, self.nside)  # type: ignore[attr-defined]
        # v2
#        self.alert_supplier.set_healpix(self.nside, self.healpix, self.time,
#            self.with_history)

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

        # Setup ingesters
        ing_hdlr = ChainedIngestionHandler(
            self.context,
            self.shaper,
            self.directives,
            updates_buffer,
            run_id,
            tier=0,
            logger=logger,
            database=self.database,
            trace_id={"alertconsumer": self._trace_id},
            compiler_opts=self.compiler_opts,
        )
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

            # Iterate over alerts
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
                                },
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
                        alertpix = hp.pixelfunc.ang2pix(hp.npix2nside(len(self.healpix_pvalues)), theta, phi, nest=True)
                        alert_pvalue = self.healpix_pvalues[alertpix]
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
                                {"alert": alert.id, "healpix": {**self.healpix_log, 'pvalue': alert_pvalue}},
#                                {"alert": alert.id, "healpix": {'pvalue':alert_pvalue}},
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

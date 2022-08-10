#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-ZTF/ampel/ztf/t0/ZTFAlertStreamController.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                07.08.2020
# Last Modified Date:  07.08.2020
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

import asyncio, copy, logging
from collections import Counter
from typing import Any
from collections.abc import Sequence

from ampel.abstract.AbsProcessController import AbsProcessController
from ampel.secret.AmpelVault import AmpelVault
from ampel.config.AmpelConfig import AmpelConfig
from ampel.core.AmpelContext import AmpelContext
from ampel.model.ProcessModel import ProcessModel
from ampel.model.UnitModel import UnitModel
from ampel.util import concurrent


log = logging.getLogger(__name__)


class ZTFAlertStreamController(AbsProcessController):

    priority: str = "default"
    multiplier: int = 1

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._scale_event: None | asyncio.Event = None
        self.update(self.config, self.vault, self.processes)

    def update(self,
        config: AmpelConfig,
        secrets: None | AmpelVault,
        processes: Sequence[ProcessModel],
    ) -> None:
        self.config = config
        self.processes = processes
        self.secrets = secrets
        self._process = self.merge_processes(list(self.processes))

    @staticmethod
    def merge_processes(processes: list[ProcessModel]) -> ProcessModel:
        assert len(processes) > 0
        process = copy.deepcopy(processes[0])

        assert process.active
        assert process.processor.unit == "AlertConsumer", "Lead process is an AlertConsumer"
        assert isinstance(process.processor.config, dict)

        def strip(config):
            """Remove AlertConsumer config keys will be changed or merged"""
            return {
                k: v for k, v in config.items()
                if k not in {"process_name", "publish_stats", "directives"}
            } if config else {}

        for pm in processes[1:]:
            # ensure that trailing AlertConsumer configs are compatible
            assert pm.active
            assert process.controller.config == pm.controller.config
            assert process.controller.override == pm.controller.override
            assert isinstance(pm.processor.config, dict)
            assert process.processor.unit == pm.processor.unit, "All processes are AlertConsumers"
            assert strip(process.processor.config) == strip(pm.processor.config), "AlertConsumer configs are compatible"
            assert strip(process.processor.override) == strip(pm.processor.override), "AlertConsumer overrides are compatible"
            process.processor.config["directives"] += pm.processor.config["directives"]
        
        process.name = Counter([proc.name.split('|')[-1] for proc in processes]).most_common(1)[0][0]

        return process

    def stop(self, name: None | str = None) -> None:
        """Stop scheduling new processes."""
        assert name is None
        self._process.active = False
        self.multiplier = 0
        if self._scale_event:
            self._scale_event.set()

    def scale(self, name: None | str = None, multiplier: int = 1) -> None:
        if multiplier < 1:
            raise ValueError("multiplier must be nonnegative")
        assert self._scale_event
        self.multiplier = multiplier
        self._scale_event.set()

    async def run(self) -> Sequence[bool]:
        """
        Keep `self.multiplier` instances of this process alive until:
          
          * they all return 0/False, or
          * the task is cancelend,
        
        whichever comes first.
        """
        assert self._scale_event is None, "run() is not reentrant"
        self._scale_event = asyncio.Event()

        def launch() -> asyncio.Task:
            counter = AbsProcessController.process_count.labels(self._process.tier, self._process.name)
            t = self.run_mp_process(
                self.config.get(),
                self.secrets,
                self._process.dict(),
            )
            counter.inc()
            t.add_done_callback(lambda t: counter.dec())
            return t
        assert self._process.active
        pending = {launch() for _ in range(self.multiplier)}
        pending.add(asyncio.create_task(self._scale_event.wait(), name="scale"))
        done: set[asyncio.Task] = set()
        try:
            while self._process.active and len(pending) > 1:
                try:
                    done, pending = await asyncio.wait( # type: ignore[assignment]
                        pending, return_when="FIRST_COMPLETED"
                    )
                    for task in list(done):
                        if task.get_name() == "scale":
                            if self._scale_event.is_set():
                                log.info(f"scale {len(pending)} -> {self.multiplier}")
                                # scale down
                                to_kill = {pending.pop() for _ in range(len(pending) - self.multiplier)}
                                for t in to_kill:
                                    t.cancel()
                                await asyncio.gather(*to_kill, return_exceptions=True)
                                done.update(to_kill)
                                # scale up
                                for _ in range(self.multiplier - len(pending)):
                                    pending.add(launch())
                                self._scale_event.clear()
                            pending.add(asyncio.create_task(self._scale_event.wait(), name="scale"))
                        else:
                            if (exc := task.exception()):
                                AbsProcessController.process_exceptions.labels(self._process.tier, self._process.name).inc()
                                log.warn("AlertConsumer failed", exc_info=exc)
                                await asyncio.sleep(10)
                            # start a fresh replica for each processor that
                            # returned True. NB: +1 for scale wait task
                            if (task.exception() or task.result()) and len(pending) < self.multiplier + 1:
                                pending.add(launch())
                except Exception:
                    for t in pending:
                        t.cancel()
                    break
        finally:
            # force scale future to come due
            self._scale_event.set()
            tasks = list(done.union(pending))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for t, r in zip(tasks, results) if t.get_name() != "scale"]

    @staticmethod
    @concurrent.process(timeout=60)
    def run_mp_process(
        config: dict[str, Any],
        secrets: None | AmpelVault,
        p: dict[str, Any],
    ) -> bool:

        try:
            import setproctitle # type: ignore
            setproctitle.setproctitle(f"ampel.t{p['tier']}.{p['name']}")
        except Exception:
            ...

        from ampel.alert.AlertConsumer import AlertConsumer

        # Create new context with frozen config
        context = AmpelContext.load(
            config=config, vault=secrets, freeze_config=True,
        )

        processor = context.loader.new_context_unit(
            model=UnitModel(**p["processor"]),
            context=context,
            sub_type=AlertConsumer,
            process_name = p["name"],
        )

        processor.run()

        return True

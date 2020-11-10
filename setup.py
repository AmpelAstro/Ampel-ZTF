#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : Ampel-ZTF/setup.py
# License           : BSD-3-Clause
# Author            : jvs
# Date              : Undefined
# Last Modified Date: 28.01.2020
# Last Modified By  : vb <vbrinnel@physik.hu-berlin.de>

from setuptools import setup, find_namespace_packages

setup(
	name='ampel-ztf',
	version='0.7.0',
	packages=find_namespace_packages(),
	package_data = {
		'conf': [
			'*.json', '**/*.json', '**/**/*.json',
			'*.yaml', '**/*.yaml', '**/**/*.yaml',
			'*.yml', '**/*.yml', '**/**/*.yml'
		]
	},
	install_requires = [
		"ampel-core",
		"ampel-interface",
		"ampel-alerts",
		"ampel-photometry",
		"confluent-kafka",
		"psycopg2-binary",
		"sqlalchemy",
	],
	extras_require = {
		"testing": [
			"pytest",
			"pytest-timeout",
			"pytest-asyncio",
			"mongomock",
			"sqlalchemy-stubs",
		]
	},
	entry_points = {
		'console_scripts': {
			'ampel-ztf-alertprocessor = ampel.ztf.t0.run:run_alertprocessor',
			'ampel-ztf-archive-consumer-groups = ampel.ztf.archive.ArchiveDB:consumer_groups_command',
		},
		'ampel_resources': [
			'archive = ampel.ztf.archive.resources:ArchiveDBURI'
		]
	}
)

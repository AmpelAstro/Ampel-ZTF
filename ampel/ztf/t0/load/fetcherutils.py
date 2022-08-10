#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/ztf/t0/load/fetcherutils.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                Unspecified
# Last Modified Date:  14.11.2018
# Last Modified By:    valery brinnel <firstname.lastname@gmail.com>

# type: ignore[import]
# pylint: disable=bad-builtin
def archive_topic():

	from ampel.ztf.t0.load.AllConsumingConsumer import AllConsumingConsumer
	from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
	import itertools, tarfile, time, os, pwd, grp, uuid, socket, fastavro, io

	parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--broker", type=str, default="epyc.astro.washington.edu:9092")
	parser.add_argument("--strip-cutouts", action="store_true", default=False)
	parser.add_argument("topic", type=str)
	parser.add_argument("outfile", type=str)

	opts = parser.parse_args()

	consumer = AllConsumingConsumer(
		opts.broker, topics=[opts.topic], timeout=20, **{'group.id':uuid.uuid1()}
	)

	def trim_alert(payload):
		reader = fastavro.reader(io.BytesIO(payload))
		schema = reader.writer_schema
		alert = next(reader)

		candid = alert['candid']
		# remove cutouts to save space
		if opts.strip_cutouts:
			for k in list(alert.keys()):
				if k.startswith('cutout'):
					del alert[k]
			with io.BytesIO() as out:
				fastavro.writer(out, schema, [alert])
				payload = out.getvalue()

		return candid, payload

	uid = pwd.getpwuid(os.geteuid()).pw_name
	gid = grp.getgrgid(os.getegid()).gr_name

	with tarfile.open(opts.outfile, 'w:gz') as archive:
		t0 = time.time()
		num = 0
		num_bytes = 0
		for message in consumer:
			candid, payload = trim_alert(message.value())
			ti = tarfile.TarInfo('{}/{}.avro'.format(opts.topic, candid))
			ti.size = len(payload)
			ti.mtime = time.time()
			ti.uid = os.geteuid()
			ti.uname = uid
			ti.gid = os.getegid()
			ti.gname = gid
			archive.addfile(ti, io.BytesIO(payload))
			num += 1
			num_bytes += len(message.value())
			if num % 1000 == 0:
				# consumer.commit_offsets()
				elapsed = time.time()-t0
				print('{} messages in {:.1f} seconds ({:.1f}/s, {:.2f} Mbps)'.format(
					num, elapsed, num/elapsed, num_bytes*8/2.**20/elapsed)
				)


def list_kafka():
	
	from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
	from pykafka import KafkaClient

	parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
	parser.add_argument("--broker", type=str, default="epyc.astro.washington.edu:9092")
	opts = parser.parse_args()
	client = KafkaClient(opts.broker)

	for name in reversed(sorted(client.topics)):
		print(name.decode())
		continue
		topic = client.topics[name]

		num = 0
		for p in topic.partitions.values():
			num += p.latest_available_offset() - p.earliest_available_offset()
		print('{}: {} messages'.format(name, num))
	# print(client.topics.keys())

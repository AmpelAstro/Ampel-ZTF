#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/ztf/t3/complement/TNSReports.py
# License:             BSD-3-Clause
# Author:              Jakob van Santen <jakob.van.santen@desy.de>
# Date:                03.11.2020
# Date:                10.03.2021
# Last Modified By:    Jakob van Santen <jakob.van.santen@desy.de>

from ampel.ztf.t3.complement.TNSNames import TNSNames


class TNSReports(TNSNames):
    """
    Add TNS reports from catalogmatch mirror
    """

    include_report = True
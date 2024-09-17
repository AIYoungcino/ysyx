#!/usr/bin/env python3
# DESCRIPTION: Verilator: Verilog Test driver/expect definition
#
# Copyright 2024 by Wilson Snyder. This program is free software; you
# can redistribute it and/or modify it under the terms of either the GNU
# Lesser General Public License Version 3 or the Perl Artistic License
# Version 2.0.
# SPDX-License-Identifier: LGPL-3.0-only OR Artistic-2.0

import vltest_bootstrap

test.scenarios('simulator')

test.compile(
    fails=test.vlt_all,
    # Used to be %Error: t/t_order_wireloop.v:\d+: Wire inputs its own output, creating circular logic .wire x=x.
    # However we no longer gate optimize this
    # Can't use expect_filename here as unstable output
    expect=
    r"%Warning-UNOPTFLAT: t/t_order_wireloop.v:\d+:\d+: Signal unoptimizable: Circular combinational logic: \'bar\'"
)

test.passes()

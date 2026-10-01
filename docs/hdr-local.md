# Local HDR configuration

`sdr_max_luminance` defaults to `0` in this branch, enabling automatic SDR white:
the monitor's EDID MaxFALL, rounded and limited to 200–500 cd/m², or 203 cd/m²
when MaxFALL is unavailable. An explicit `max_avg_luminance` override takes
precedence over the EDID value. A positive `sdr_max_luminance` sets SDR white
explicitly.

This intentionally changes the upstream default of 80 cd/m². Set
`sdr_max_luminance` to `80` to retain that behavior. If this change is submitted
upstream, the monitor documentation needs a corresponding wiki change.

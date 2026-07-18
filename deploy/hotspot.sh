#!/bin/bash
# Turn the box into a Wi-Fi access point so every phone is its screen.
# The demo's "unplug the router" beat: phones join BUGOUT-BOX directly.
# Run on the Pi. Reverts on reboot unless made permanent.
set -e
SSID="${1:-BUGOUT-BOX}"
PASS="${2:-shelter72}"
sudo nmcli device wifi hotspot ifname wlan0 ssid "$SSID" password "$PASS"
echo "AP up: SSID=$SSID  pass=$PASS"
echo "Dashboard: http://10.42.0.1:8880  (NM default AP subnet)"

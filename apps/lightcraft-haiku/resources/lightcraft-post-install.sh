#!/bin/sh
app=$(findpaths -p "$0" B_FIND_PATH_APPS_DIRECTORY LightCraft 2>/dev/null)
[ -n "$app" ] && [ -e "$app" ] && mimeset -a -f "$app"
exit 0

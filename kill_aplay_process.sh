#!/bin/bash

# Check if any previous aplay process is running and kill it
echo "Stopping any ongoing aplay processes..."
pkill aplay

# Play the new sound file in the background
echo "Playing new sound: $1"
aplay -q "$1" &

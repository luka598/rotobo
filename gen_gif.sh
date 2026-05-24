#!/bin/bash

FPS=10

ffmpeg -framerate $FPS -i ignore_img/%03d.png -vf palettegen palette.png

ffmpeg -framerate $FPS -i ignore_img/%03d.png -i palette.png -lavfi paletteuse output.gif

rm palette.png

OpenCV Drawing App (Improved Version)

Files
- main.py
- config.py
- image_loader.py
- state.py
- drawing_tools.py
- mouse_handler.py
- app.py

Install
1) python -m pip install opencv-python numpy

Run
2) python main.py --path logo1.bmp

Example
3) python main.py --path logo1.bmp --output result.png

Keys
- r : rectangle mode
- l : line mode
- c : crop selected rectangle (rectangle mode only)
- s : save current image
- h : help
- ESC : exit

Notes
- In line mode, the line is committed to the image when the mouse button is released.
- Crop works only in rectangle mode.
- Saved output file name can be changed with --output.

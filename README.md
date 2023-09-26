# ReceiptSeparator
Separate and compress images of goup scans of receipts into individual ones
Code adapted from the Interactive Grabcut repo from Jason Y. Zhang : https://github.com/jasonyzhang/interactive_grabcut

This small code loads pictures of Receipts in a folder (potentially with several receipts on a same images), and then applies the GrabCut algorithm in an interactive fashion to extract individual, compressed images of the receipts. 

Upon loading an image, it will show a resized version of it. The user only needs to draw a bounding box around the group of elements and press 'q' to run the algorithm. He/she will then be presented with the result, which can be adapted by telling the algorithm where it did mistakes. Pressing "b" or "f" and drawing a bounding box will respectively tell the algorithm where background or foreground is, and pressing "g" will rerun th algorithm. When good, press q to save independent receipts and go to the next image. 

For usage and arguments, run "python3 ReceiptSeparator.py -h"

Dependencies: python3, openCV and numpy. 

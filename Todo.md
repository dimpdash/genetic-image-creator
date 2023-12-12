# TODO
[ ] Optimise sum pipeline
    - created two shaders 
        - one that convers textures to a buffer
        - one that sums the buffer in blocks
    - then just summing the remaining blocks on cpu
    - still to do
        * for some reason only two images are being processed but it expects 100
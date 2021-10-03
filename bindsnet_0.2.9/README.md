# Repository contents at this directory level
This folder contains [BindsNET](https://github.com/BindsNET/bindsnet) version 0.2.9. This folder has been included, as well as the frozen versions of the dependencies to ensure that the implemented code continues to work even if there are updates. 

However, it is recommended to use the most updated version of BindsNET and modify the code as necessary to work with that version. To use newer versions of the dependencies, modify the *setup.py* file, changing the *==* to the desired comparison symbol (*<, <=, >, <=, <=*).

To install BindsNET run the following command inside the *bindsnet_0.2.9* folder:
```bash
pip install [-e] .
```
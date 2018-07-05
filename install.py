import os
print("Installing the main library")
packages = ["pillow","opencv","opencv-contrib-python","numpy"]
for package in packages:
	os.system("sudo python3 -m pip install "+str(package))
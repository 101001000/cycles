rm lib/linux_x64/portablert
rm prt
ln -s ../portableRT/build/install prt
ln -s ../../prt lib/linux_x64/portablert
mv lib/linux_x64/embree lib/linux_x64/embree2
cp -r ../../embree lib/linux_x64/embree
mv lib/linux_x64/dpcpp lib/linux_x64/dpcpp2
cp -r ../../dpcpp lib/linux_x64/dpcpp
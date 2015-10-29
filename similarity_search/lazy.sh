cmake .;
make

cd ../python_bindings

sudo rm /usr/local/lib/python2.7/dist-packages/nmslib.so
rm nmslib.so -f

sudo make install

cd ../similarity_search

python lazy_test.py

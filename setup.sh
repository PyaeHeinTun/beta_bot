apt-get update

#Ta-Lib Install
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure
make
make install
pip install ta-lib
rm -rf ta-lib-0.4.0-src.tar.gz
rm -rf ta-lib

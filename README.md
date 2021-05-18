# Usage:

## Unpacking the data
```shell script
cat resources/data/x* > resources/data/data.zip
unzip resources/data/data.zip -d resources/data
rm -r resources/data/x*
rm resources/data/data.zip 
```

## Launch 
```shell script
python start.py
```
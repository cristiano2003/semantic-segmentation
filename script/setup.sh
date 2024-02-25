# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.usercontent.google.com/download?id=1CcClO9ZWsDEszn55A_T4CbqmlMLzL_46&export=download&authuser=0&confirm=t&uuid=a5a79d0c-1775-4672-8cd7-b068a94da115&at=APZUnTWWaMh06-Mf5egSyDO2cuUS%3A1708850269417' > "data.zip"
  unzip -q "data.zip" -d "data"
fi


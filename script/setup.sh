# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.google.com/file/d/1CcClO9ZWsDEszn55A_T4CbqmlMLzL_46/view?usp=share_link' > "data.zip"
  unzip -q "data.zip" -d "data"
fi


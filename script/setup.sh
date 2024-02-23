# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L "https://www.kaggle.com/datasets/lqdisme/cityscapes/download?datasetVersionNumber=1" > "data.zip"
  unzip -q "data.zip" -d "data"
fi


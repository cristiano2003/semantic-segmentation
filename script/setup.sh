# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?id=1MPlji4PXxA-b4C_b6YRMLF5s_n_TL5fP&export=download" > "data.zip"
  unzip -q "data.zip" -d "data"
fi


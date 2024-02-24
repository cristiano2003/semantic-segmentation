# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.usercontent.google.com/download?id=1MPlji4PXxA-b4C_b6YRMLF5s_n_TL5fP&export=download&authuser=0&confirm=t&uuid=a576b4ff-bdef-4726-a726-99861f72d055&at=APZUnTWH7ZTbMkHk_SPaW2LX-YwF:1708769132683' > "data.zip"
  unzip -q "data.zip" -d "data"
fi


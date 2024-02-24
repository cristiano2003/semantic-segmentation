# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.usercontent.google.com/download?id=1MPlji4PXxA-b4C_b6YRMLF5s_n_TL5fP&export=download&authuser=0&confirm=t&uuid=4bfc62bf-474f-463a-ab72-47bb3d6273bf&at=APZUnTWormyg31XeUn40AJajWnZ5%3A1708768390747' > "data.zip"
  unzip -q "data.zip" -d "data"
fi


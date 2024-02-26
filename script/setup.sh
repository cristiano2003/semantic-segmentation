# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.usercontent.google.com/download?id=1CcClO9ZWsDEszn55A_T4CbqmlMLzL_46&export=download&authuser=0&confirm=t&uuid=d40e1212-60cf-4190-bba9-0963fe5e7e6e&at=APZUnTVI5vYR1kqfNYg2_zC1Aaam%3A1708920832696' > "data.zip"
  unzip -q "data.zip" -d ""
fi


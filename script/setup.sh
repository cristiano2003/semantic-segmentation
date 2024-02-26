# install package
pip install -e .

curl for dataset, download if not already downloaded
if [ -d "data" ]; then
  echo "data already downloaded"
else
  echo "data not downloaded, downloading now"
  curl -L 'https://drive.usercontent.google.com/download?id=1x8o4juYBgwxXbRt_EuaHCxvCeOLzUcws&export=download&authuser=0&confirm=t&uuid=949819b8-9325-4611-8a81-a10bfdca79dd&at=APZUnTUw2JZjwKAWr-d-VYxfE8yU%3A1708921918366' > "data.zip"
  unzip -q "data.zip" -d ""
fi


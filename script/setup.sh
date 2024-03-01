# install package
pip install -e .

# curl for dataset, download if not already downloaded
# if [ -d "data" ]; then
#   echo "data already downloaded"
# else
#   echo "data not downloaded, downloading now"
#   curl -L 'https://drive.usercontent.google.com/download?id=1cBQy7BgvdhZcSVbRBwXS8tfVqj7NIiOs&export=download&confirm=t&uuid=556de15a-cbc0-40f9-ac70-ee1a5ffe58e6' > "model.ckpt.zip"
#   unzip -q "model.ckpt.zip" -d "checkpoints/model/"
# fi


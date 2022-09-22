# To entry in conda env
conda activate butia_ws

# To install ffmpeg library
echo "Installing ffmpeg library..."
apt install ffmpeg
echo "Done..." 

# To install necessary libraries to run the package
echo "Installing python libraries..." 
pip3 install -r requirements.txt
echo "Done..." 

# To install python modules
echo "Installing python modules..."
python3 -c "import nltk;nltk.download('stopwords');nltk.download('punkt')"
echo "Done."
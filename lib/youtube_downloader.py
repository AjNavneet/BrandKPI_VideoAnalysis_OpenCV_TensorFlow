from pytube import YouTube  # Import the YouTube class from the PyTube library

# Set the path where the video will be saved
SAVE_PATH = "./"

# The YouTube video link to be downloaded
link = "https://www.youtube.com/watch?v=xWOoBJUqlbI"

try:
    yt = YouTube(link)  # Create a YouTube object for the given video link
except:
    print("Connection Error")  # Handle the exception

yt.streams.first().download()  # Download the first stream (usually the highest quality)

print('Task Completed!')  

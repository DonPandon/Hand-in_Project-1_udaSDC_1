import moviepy.editor as mp
clip = mp.VideoFileClip("tbay_testing_3.mp4")
clip_resized = clip.resize(height=540)
clip_resized.write_videofile("movie_resized.mp4", audio=False)
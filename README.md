#**Finding Lane Lines on the Road** 

##Prepared by Daniel Garcia Gonzalez on February 25th, 2017

###I prepared this write up as part of my revisited version of project 1: video pipeline.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./challenge_Mask.jpg "Challenge Mask"
---

### Reflection

###0. Introduction

Jupyter wasn't working, Ubuntu was a stuborn child on me when trying to install python and cv2, cv2 saving videos of 6 kb, divisions by zero, loosing float points, and, even worst, an angry wife...

I switched to PyCharm to work on this project, so I had to modify the delivery of it. To review my code, please open the py file called **"Project1_DanielGG_mySolution.py"**. The whole repository might have residues of the original Udacity template, so please disregard them.

Once you go over my code, you'll notice I lay out the code in a way - I think - that is easier to come back in the future and read through the whole solution as if it was notebook. I put labels and indications for me to remember why I had to make changes, where're my functions, where are the functions given by the Udacity instructors, and such. I also include sections where I incorporate my troubleshooting code. I am used to programming since I am a kid, but I am much more visual, specially when I want to fix something.

I call my code a Patchwork because it doesn't come 100% from my brain. I went on forums, blogs, other peers, my tutor, colleges, even my best friend had to take some times from his grad studies to discuss a few things with me (not that I forced him, he just can't skip a chance to procrastinate). What you'll find that is completely mine are my notes and mental process working along the solution, notes to where I got stuck, and how I found how to solve my problems.

In the end, I was victorious. Unfortunately, the code is not perfect. There's still a lot of work to do and I am running out of time by now so, let's get started.

 I present to you my patchwork; I mean, pipeline.

###1. Course of action: What to expect in this project.

cv2 was not doing the job of saving the file correctly into a video and, checking online, it was found that cv2 has a bug that makes it almost impossible for it to save videos. The first time I handed in the video, I stored up to 300 images in a single folder and used Movie Maker to put it together as a video. Not elegant, but did the job. The problem was that it was missing the capability of creating smooth lines; instead, it was all the Hough’s' jumping around in a complete chaos. Not practical.

This new version uses moviePy to properly open, edit, and save the video for the pipeline to be complete.

The steps I followed were (after opening and initializing the video for editing):

1. The Trifecta

  1. I Converted each frame to a grayscale picture, then

  2. Smooth the frame with Gaussian Blur to find the bit change rate from the frame, and then

  3. Applied Canny Edge to take that change and covert it into individual pixels for processing.


2. Reducing the Region of Interest to what we want to work with (The ROI)

  1. Created a blank frame (the size of the input frame) and defined a small area to be the only area visible by using a Polygon. In most videos, a trapezoid did the work; but more sophisticated shapes can be used to exactly define what will be "visible" as the Region of Interest.

  2. Fused the mask to the sequence of bits from the previous step to have only the pixels from the lanes.


3. Hough's transformation

  The masked region now is transformed using the Hough's algorithm to generate lines out of the composition of points from the previous step. These lines are what gives the power to the pipeline because is what can be used as the "recognized lines" from each lane.


4. Obtaining the mean lines

  1. Squeezed any undesired dimensions that could have been leaked up to this point. They won't do any benefit to the code nor the functionality of the program.

    The Hough lines obtained are lists full of 2 points in 2-D; so, 4 points per line; so,

  2. What comes after is just obtaining the mean line derivate from the larger Hough lines, also averaged. It could have been the mean from all the lines, but with smaller lines, the mean line was barely affected. The end calculation generates a "mean set of points x1, y1, x2, y2" for both left and right lanes that are drawn using the draw line function, given by Udacity.

  3. Fused the lines and the input frame, start assembling the video and recall the next input frame.

###2. Content of the program described
My program is split in 5 major areas (not including the import section) to make it easier to identify where modifications have to be done: Variables, Mask Points, My Functions, Project Functions, and Execution.

* Variables

  In This section, I include all the variables associated to literals of my program. To make modifications on values, it's easier to track down this section than fishing them from the code.


* Mask points

  In this section, you may find the different masks tweaked for each different input video. Almost all of them used a trapezoidal polygon to do the work. However, the challenge video required an more sophisticated shape to eliminate as much noise as possible. The shape, if drawn by hand, would look like this:

<p align="center">

<img src="/challenge_Mask.jpg">

</p>

* My Functions

  In this section, you may find the functions I generated for my code to work. The code could be more modular, but that could be listed as a future improvement for now.


* Project Functions

  In this section, you may find the functions given by Udacity as part of the template project. I put them separately to avoid confusion (of myself) and to have clear which functions needed adjustment, which didn't. Hough’s did get a transformation because both of its outputs were used as part of the smoothing process.


* Execution

  In this section, you may find the code that is actually doing something. It loads the video into the stream, edits it, and saves it back as a new file.


###2. Identify potential shortcomings with your current pipeline

* One potential shortcoming would be which version of Python you are using. Some functions may work slightly different than others, or the result for calculations (specially floats) might give a somewhat different output. The very first line of code was incorporated to compensate a problem I found while troubleshooting my code: my division were done as integers and I needed float results, so a compatibility import had to be done for division.

* Another shortcoming is my layout. I do not tend to follow industrial or common layout procedures. So, experienced programmers, I’m sorry.

* When storing the output points to generate the lines and with videos with shadows, I was getting mathematical errors in my generated points. I couldn't find a more elegant way to solve this problem but to put a try - except and skip the points generating this issue and just continue with the next one. This is what makes the lines disappear if there's not enough "line to track" on the input frame.

* I am doing a lot of designations and small calculations all over the place. For now, this is "good enough" but when more sophisticated code comes, it may cause a lot of drag.

###3. Suggest possible improvements to your pipeline

* A possible improvement would be to improve my algorithm to make smooth lines. For now, I am just writing a new line out of mean values from the input lines; by watching what other people has done, it is obvious there are way better ways to do this. I see a lot of repetitive actions in my code, but I am not experienced enough to reduce it more.

* Another potential improvement could be to continue identifying better literals to implement to boost the efficiency of my output. Now, by hand, this will take me forever. I’ll guess this is where machine learning might come handy and identify the "best of all" values to use for the current conditions applied.

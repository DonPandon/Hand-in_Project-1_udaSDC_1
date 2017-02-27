#**Finding Lane Lines on the Road** 

##Prepared by Daniel Garcia Gonzalez on February 25th, 2017

###I prepared this write up as part of my revisited version of project 1: video pipeline.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
---

### Reflection

###0. Introduction

Jupyter wasn't working, Ubuntu was a stuborn child on me when trying to install python and cv2, cv2 saving videos of 6 kb, divisions by zero, loosing float points, and, even worst, an angry wife...

I switched to PyCharm to work on this project, so I had to modify the delivery of it. To review my code, please open the py file called **"Project1_DanielGG_mySolution.py"**. The whole repository might have residues of the original Udacity template, so please disregard them.

Once you go over my code, you'll notice I lay out the code in a way - I think - that is easier to come back in the future and read through the whole solution as if it was notebook. I put labels and indications for me to remember why I had to make changes, where're my functions, where are the functions given by the Udacity instructors, and such. I also include sections where I incorporate my troubleshooting code. I am used to programming since I am a kid, but I am much more visual, specially when I want to fix something.

I call my code a Patchwork because it doesn't come 100% from my brain. I went on forums, blogs, other peers, my tutor, colleges, even my best friend had to take some times from his grad studies to discuss a few things with me (not that I forced him, he just can't skip a chance to procrastinate). What you'll find that is completely mine are my notes and mental process working along the solution, notes to where I got stuck, and how I found how to solve my problems.

In the end, I was victorious. Unfortunately, the code is not perfect. There's still a lot of work to do and I am running out of time by now so, let's get started.

 I present to you my patchwork; I mean, pipeline.

###1. Course of action: How the program was tackled.

After the Hough's transformation is applied, a collection of lines is stored and generates a very interesting effect that looks like this (exagered):

<p align="center">

<img src="/HoughLines_unprocessed.jpg">

</p>

That works "fine" to understand the concept, but it won't do any good to the self driving algorythm. Reason why a more smooth line is required. Such line can be obtained with multiple statistics operations such as interpolation. My limitations in Python, however, put me in a bad place to know how to perform complex calculations with this programming language, so I decided to use a more "simple" process: a mean line.

What I call a mean line is a line that is generated out of the analysis of all the lines created by Hough's. Similar to the interpolation of points, this mean line will check the  mean values of all the points forming the lines ( Exes (X) and Wyes (Y) ) to get the final mean lines X and Y values. Positive and Negative slopes can be use to catheorize which lines represent left lane (negative) and which ones the right lane (positives). All the calculations used will work around the standard line equation:

<p align="center">
**y = m x + b**
</p>

All those lines have slopes and magnitudes that can be used to categorize them. The magnitudes will be used to identify which lines are more signitifcant (i.e. will bring the more significant change to the result), and the slopes will be used to sort the points in different groups. Definining a threshold for the slope is important to avoid situations in which we get divisions by zero, for example.

This process is not optimal due to the high dependecy on the output of Houghs, meaning that errors and noise in the pre-processed image will generate a lot of "jumping" of the new lines. Howvwer, is good enough for the scope of this project. The results can be shown in the video as a collection of frames that assemble the mean X and Y points of the mean line and draws them over the lanes:

<p align="center">

<img src="/meanLines.JPG">

</p>



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

* Maybe, instead of adjusting both X and Y values of the output mean lines, I should adjust only the Ys. Xs tend to stay at the same point.

* Another potential improvement could be to continue identifying better literals to implement to boost the efficiency of my output. Now, by hand, this will take me forever. I’ll guess this is where machine learning might come handy and identify the "best of all" values to use for the current conditions applied.

###4. Conclusion

I had a lot of learning and joy (at the end) after working on this project. It still has a lot of work to be done, but for now I feel I made a big improvement compared to what I had in previous weeks. I hope to dedicate more time to the futurte project so I can finish before the hard dead line and be able to continue to term 2. This is getting exiting.

Have an excellent day! - Daniel
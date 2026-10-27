# Speaker script — Group 10

The same text is in the notes of each slide in `BING_Objectness_Group10.pptx`.

| Speaker | Slides | Time |
|---|---|---|
| Abhinav Dileep | 1, 2, 3 | ~1:45 |
| Ayyappadas M T | 4, 5, 6 | ~2:20 |
| Hari Sankar A | 7, 8, 9 | ~2:20 |
| Parthiv M | 10, 11, 12, 13 | ~3:25 |
| Sai Kishen K M | 14, 15, 16, 17 | ~3:25 |
| **Total** | 1–17 | **~13:15** |

## Slide 1 — Abhinav Dileep (~35 s, 0:00–0:35)

Good morning. We are Group 10, and our project is "Fast Object Proposal Generation Using Handcrafted Gradient Features". It is based on the CVPR 2014 paper BING, by Cheng, Zhang, Lin and Torr. BING finds the regions of an image that are likely to contain objects, at around 300 frames per second, using only image gradients and bit operations, with no deep learning. I will introduce the problem, and then my teammates will cover our solution, our own extensions and our results.

## Slide 2 — Abhinav Dileep (~35 s, 0:35–1:10)

Why do we need object proposals? A classic detector slides a classifier over every position, every size and every shape of window. That is a hundred thousand windows or more per image, and almost all of them are background. Object proposals solve this. A cheap, class-agnostic score ranks all the windows first, so the expensive detector only has to look at the top thousand or so. BING is one of the fastest ways to do this: the paper reports 300 images per second.

## Slide 3 — Abhinav Dileep (~35 s, 1:10–1:45)

Here is our problem definition, in the box: given an image, output a ranked list of boxes that covers as many objects as possible with few boxes, in milliseconds, using only handcrafted features. The constraints: no class labels, real time, CPU only, no deep networks. We measure three things: the detection rate, meaning the share of objects found in the top N boxes; MABO, meaning how tight the boxes are; and speed. We answer where objects are, not what they are. Ayyappadas will now explain our proposed solution.

**→ Hand over to Ayyappadas M T.**

## Slide 4 — Ayyappadas M T (~50 s, 1:45–2:35)

Thank you, Abhinav. BING's key idea is simple: objects have closed outlines. Shrink any window to 8 by 8 pixels and look at its gradient strength, and a window around an object shows a ring of strong edges. A linear model with just 64 weights learns this template; on the right is the filter we learned, positive on the border and negative inside. To make it fast, filter and features are turned into bits. Beyond the paper, we made four contributions, shown in the cards. Our main novelty is a third stage that re-ranks BING's windows with extra cues. We built a complete object detector on top of our proposals. We compared our version head to head with the authors' own C++ code, which revealed two details the paper leaves out. And everything is a from-scratch, multi-threaded Python implementation.

## Slide 5 — Ayyappadas M T (~40 s, 2:35–3:15)

This is our system design, the same eight stages as in our proposal. The input image is shrunk to at most 500 pixels and converted to a colour space. We consider 36 window sizes. For each size, the whole image is resized so that every 8 by 8 patch corresponds to exactly one window. We compute the gradients, binarize them, and score every window with bit operations. Greedy suppression keeps the best spread-out windows, a per-size calibration makes scores comparable across sizes, and the top boxes come out. After that, our own stage three re-ranks them, which I will show next.

## Slide 6 — Ayyappadas M T (~50 s, 3:15–4:05)

This is our main novelty, a third stage after BING. BING ranks windows with a single cue, its 8 by 8 edge template, so some object windows end up far down the list. Our stage re-scores BING's windows with four groups of cheap, handcrafted cues, listed on the right: the colour contrast between the window and a ring around it, the edge density inside the window against the ring, the window's position, size and shape, and BING's own score. A small gradient-boosted tree model, trained on VOC trainval, combines them into a new score. Every cue is read from integral images in constant time, so there is still no deep network. On this test image, BING's top 20 windows find one of the three objects; after re-ranking they find all three. Hari will now explain the bit-level trick and our experimental setup.

**→ Hand over to Hari Sankar A.**

## Slide 7 — Hari Sankar A (~45 s, 4:05–4:50)

Thank you, Ayyappadas. On the left, the top row shows the gradient maps of a coffee-cup image for three window sizes. The bottom row shows the four most significant bit planes of one map. The trick is on the right. The 64-weight filter is approximated by a weighted sum of plus-and-minus-one vectors. Each gradient value is approximated by its top four bits. Each set of 64 bits fits in one 64-bit integer, so a dot product becomes an AND followed by a POPCOUNT. That is 12 popcount operations per window instead of 64 multiply-adds. Our unit tests confirm that the bitwise score exactly equals the score computed with floats.

## Slide 8 — Hari Sankar A (~40 s, 4:50–5:30)

For evaluation we used three datasets. The main one is PASCAL VOC 2007, the paper's own benchmark: we train on 5,011 images and test on 4,952 images with over 12,000 objects. We also generated a synthetic benchmark of 900 images with cluttered backgrounds, and we annotated 58 objects in 10 real photographs that are never used for training. Every method is scored with the same evaluator. As baselines we used the authors' own C++ code, Selective Search, EdgeBoxes, sliding windows and random boxes. For the detector, we report the standard VOC mean average precision.

## Slide 9 — Hari Sankar A (~55 s, 5:30–6:25)

One design choice is the colour space in which the gradients are computed. Our teacher asked us to compare RGB, HSV and grayscale, so we trained one BING model per colour space on VOC and tested each on the full test set. For colour images the gradient is taken in every channel and the strongest one is kept. On VOC, HSV is slightly best: 94.4 percent of objects in the top 1000, against 94.0 percent for RGB and 93.2 percent for gray. Gray misses edges that differ only in colour, so it loses most on small objects. But the winner depends on the data: RGB is best on our synthetic scenes, and gray transfers best from synthetic training to real photos. Merging the three lists, as the paper's BING-Diversified does, finds 98.4 percent of objects at 5000 boxes, the highest of all, but it is worse at small budgets. We kept RGB as our default, as in the paper. Parthiv will now present our results.

**→ Hand over to Parthiv M.**

## Slide 10 — Parthiv M (~55 s, 6:25–7:20)

Thank you, Hari. This is our main result, on PASCAL VOC 2007. Our final BING finds 94.0% of all objects in its top 1000 proposals. The authors' own C++ code, on the same images with the same evaluator, finds 93.8%. So our rebuild matches the original. When we use the authors' own evaluator, their code scores 96.0 percent, against 96.2 percent reported in the paper, so the paper's result reproduces. We also repeated the paper's generalisation test: trained on only six classes, BING still finds 91.1 percent of the objects of the other fourteen classes, close to 93.3 percent for the model trained on all twenty. So it learns generic objectness. The dashed blue line is our first version. It stopped at 75 percent, and I will explain why in a moment.

## Slide 11 — Parthiv M (~45 s, 7:20–8:05)

This table compares BING with other methods on all 4,952 VOC test images, all timed on the same machine. BING finds the most objects in the top 1000: 94.0 percent against 89.2 percent for Selective Search and 91.6 percent for EdgeBoxes, and it is about 15 times faster than EdgeBoxes and 77 times faster than Selective Search. But both draw tighter boxes, which shows in the MABO column. Compared with the authors' code, our version has the same accuracy on VOC, generalises better to real photos when trained only on synthetic data, and is faster on one thread, 98 against 76 images per second.

## Slide 12 — Parthiv M (~50 s, 8:05–8:55)

Our first version followed the paper's text exactly and reached only 75 percent on VOC. By running the authors' released code, we found two details that are not in the paper. First, window sizes: the paper lists 10 to 320 pixels, but the code uses 16 to 512 and clips large windows to the image, and VOC has many large objects. Second, suppression: we kept only local peaks, but the code uses greedy suppression, which keeps more good windows. As the table shows, each helps, and together they close the gap. Below is recall per VOC class: thin objects like bottles are hardest; cats and dogs are easiest.

## Slide 13 — Parthiv M (~55 s, 8:55–9:50)

Next, the results of our novelty on the full VOC 2007 test set. With re-ranking, the top 100 windows find 71.0 percent of all objects instead of 67.2 percent, and the top 1000 find 53.3 percent of small objects instead of 39.3 percent. The curve on the left is above BING's at every budget. The bars on the right show which cues matter: removing edge density hurts most, especially for small objects. The cost is speed: 17 instead of 10 milliseconds per image on one thread, which is still about 46 times faster than Selective Search. We also tried snapping box edges to strong gradients and adding finer window sizes; neither helped on the full test set, so we report them as negative results. Sai Kishen will now show the detector we built on our proposals, and conclude.

**→ Hand over to Sai Kishen K M.**

## Slide 14 — Sai Kishen K M (~80 s, 9:50–11:10)

Thank you, Parthiv. Proposals exist to feed a detector, so we built one, in two versions. Both use one linear classifier per VOC class, trained on VOC trainval with hard-negative mining. The first describes each window with HOG and a colour histogram, fully handcrafted. The second takes each window's average activations from a ResNet-50 pretrained on ImageNet, and adds box regression to tighten the boxes. With HOG the best result is 7.4 percent mean average precision. With CNN features it is 19.1 percent, again with our re-ranked top 100 windows. The chart shows the CNN version. At 100 windows, BING's boxes give 18.8 percent against 13.8 percent for random boxes and only 0.6 percent for sliding windows, so proposals matter. With 1000 windows, random boxes catch up and reach 16.0 percent. The plot on the right explains why: at an overlap of 0.7, BING's top 1000 cover only 30 percent of objects, against 43 percent for random boxes. A detector needs tight boxes, which matches what Hosang and colleagues found in their 2016 study of proposals.

## Slide 15 — Sai Kishen K M (~40 s, 11:10–11:50)

Here is the CNN detector on a few photos. It finds the motorbike, the brown horse and its rider, the astronaut and the cameraman, and names each with one of the twenty VOC classes. It also shows the limits: it misses the white horse, it only knows the VOC classes, so a cup or a phone gets no label, and its boxes can be loose, like the motorbike box that cuts off the front wheel. In short, BING says where objects are, and the detector says what they are.

## Slide 16 — Sai Kishen K M (~40 s, 11:50–12:30)

Our ablation studies show what matters. The simple minus-one, zero, one gradient mask works as well as Sobel. RGB is the best single colour space. Just two bits of gradient are enough. And because the model has only 64 weights, even 10 training images give good results. On the right is speed. With four threads, each working on its own image, our version reaches 294 images per second, close to the authors' C++ code at 196. The main limitation is that BING's boxes are coarse, because the window sizes are powers of two and the stride is 8 pixels.

## Slide 17 — Sai Kishen K M (~45 s, 12:30–13:15)

To conclude: simple gradients, a 64-weight linear model and bit operations find 94 percent of objects on PASCAL VOC, matching the authors' own code. Our novelty, multi-cue re-ranking, finds 71.0 percent of objects in the top 100 and raises small-object recall from 39.3 percent to 53.3 percent. Our detector reaches 19.1 percent mean average precision with CNN features, beats sliding windows by a wide margin, and shows that BING's loose boxes are what limits detection. In future we would like to make the boxes tighter and fine-tune the CNN features. Thank you. We are happy to take questions, and we can show a live webcam demo.

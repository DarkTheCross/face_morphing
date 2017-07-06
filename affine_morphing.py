import numpy as np
import dlib
import cv2
import sys

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '|' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def findIndexInLandmark(npArray, val):
    for i in range(0, npArray.shape[0]):
        if npArray[i,0] == int(val[0]) and npArray[i,1] == int(val[1]):
            return i
    return -1

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def getTrianleImage(imgFrom, tr1, tr2):
    imgTo = np.zeros_like(imgFrom)
    r1 = cv2.boundingRect(tr1)
    r2 = cv2.boundingRect(tr2)
    tri1Cropped = []
    tri2Cropped = []
    for i in xrange(0, 3):
        tri1Cropped.append(((tr1[0][i][0] - r1[0]),(tr1[0][i][1] - r1[1])))
        tri2Cropped.append(((tr2[0][i][0] - r2[0]),(tr2[0][i][1] - r2[1])))
    img1Cropped = imgFrom[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
    img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);
    img2Cropped = img2Cropped * mask
    imgTo[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = imgTo[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    imgTo[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = imgTo[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped
    return imgTo

def imgAddNoZero(img1, img2):
    img3 = img1 + img2
    img3OverlayPixels = np.transpose(np.nonzero(img1 * img2))
    for i in img3OverlayPixels:
        img3[i[0]][i[1]] = img1[i[0]][i[1]]/2 + img2[i[0]][i[1]]/2
    return img3

def getLandmarks(imgPath, dlibDetector, dlibPredictor):
    img = cv2.imread(imgPath)
    dets = dlibDetector(img, 1)
    if len(dets) != 1:
        print "Face num not 1!"
        return []
    shape = dlibPredictor(img, dets[0])
    keyPoints = []
    for i in range(0,shape.num_parts):
        keyPoints.append((shape.part(i).x, shape.part(i).y))
    A = img.shape
    keyPoints.append((0, 0))
    keyPoints.append((0, A[0]-1))
    keyPoints.append((A[1]-1, 0))
    keyPoints.append((A[1]-1, A[0]-1))
    keyPoints.append((0, (A[0]-1)/2))
    keyPoints.append(((A[1]-1)/2, 0))
    keyPoints.append((A[1]-1, (A[0]-1)/2))
    keyPoints.append(((A[1]-1)/2, A[0]-1))
    return keyPoints

def generateMorphingImage(img1, img2, lm1, lm2, alpha):
    lm3 = lm1*alpha + lm2*(1-alpha)
    lm3 = lm3.astype("uint32")
    for i in range(0, len(lm3)):
        lm3[i] = (lm3[i, 0], lm3[i, 1])
    A = img1.shape
    rect = (0, 0, A[1], A[0])
    subdiv = cv2.Subdiv2D(rect)
    for i in lm3:
        subdiv.insert((i[0], i[1]))
    triangles3 = subdiv.getTriangleList()
    triangleAsId = [];
    for i in triangles3:
        pt1 = (i[0], i[1])
        pt2 = (i[2], i[3])
        pt3 = (i[4], i[5])
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            triangleAsId.append([findIndexInLandmark(lm3, pt1), findIndexInLandmark(lm3, pt2), findIndexInLandmark(lm3, pt3)])
    #Test affine transform
    imgMorphingFrom1 = np.zeros_like(img1)
    imgMorphingFrom2 = np.zeros_like(img2)
    for i in range(0, len(triangleAsId)):
        tr11 = np.float32([[[lm1[triangleAsId[i][0]][0], lm1[triangleAsId[i][0]][1]],[lm1[triangleAsId[i][1]][0], lm1[triangleAsId[i][1]][1]],[lm1[triangleAsId[i][2]][0],lm1[triangleAsId[i][2]][1]]]])
        tr33 = np.float32([[[lm3[triangleAsId[i][0]][0], lm3[triangleAsId[i][0]][1]],[lm3[triangleAsId[i][1]][0], lm3[triangleAsId[i][1]][1]],[lm3[triangleAsId[i][2]][0],lm3[triangleAsId[i][2]][1]]]])
        imgMorphingFrom1 = imgAddNoZero(imgMorphingFrom1, getTrianleImage(img1, tr11, tr33))
        tr22 = np.float32([[[lm2[triangleAsId[i][0]][0], lm2[triangleAsId[i][0]][1]],[lm2[triangleAsId[i][1]][0], lm2[triangleAsId[i][1]][1]],[lm2[triangleAsId[i][2]][0],lm2[triangleAsId[i][2]][1]]]])
        imgMorphingFrom2 = imgAddNoZero(imgMorphingFrom2, getTrianleImage(img2, tr22, tr33))
    imgResult = cv2.addWeighted(imgMorphingFrom1, alpha, imgMorphingFrom2, 1-alpha, 0)
    return imgResult

predictor_path = "shape_predictor_68_face_landmarks.dat"
face1path = "0.png"
face2path = "1.png"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

landmark1 = getLandmarks(face1path, detector, predictor)
landmark2 = getLandmarks(face2path, detector, predictor)
if len(landmark1)==0 or len(landmark2)==0:
    print "Error!"

L1 = np.array(landmark1)
L2 = np.array(landmark2)

face1 = cv2.imread(face1path)
face2 = cv2.imread(face2path)

for i in range(1,10):
    img3 = generateMorphingImage(face1, face2, L1, L2, np.float32(i)/10)
    printProgress(i, 9, barLength = 50)
    cv2.imwrite("affine/img"+str(i)+".png", img3)

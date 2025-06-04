import cv2
import numpy as np
import time
from collections import defaultdict
import math
from picamera2 import Picamera2


class LaneStateMachine:
    def __init__(self, stabilityThreshold=5):
        self.state = "No Lane Detected"
        self.counters = defaultdict(int)
        self.threshold = stabilityThreshold

    def update(self, newState):
        for key in self.counters:
            if key != newState:
                self.counters[key] = 0
        self.counters[newState] += 1
        if self.counters[newState] >= self.threshold:
            if newState != self.state:
                self.state = newState
                for key in self.counters:
                    self.counters[key] = 0
        return self.state


def calculateSlopeAngle(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return 90
    slope = (y2 - y1) / (x2 - x1)
    return math.degrees(math.atan(abs(slope)))


def detectLanes(frame, cropMarginPercent):
    height, width = frame.shape[:2]
    cropMarginX = int(width * cropMarginPercent)
    cropMarginY = int(height * 0.3)
    roi = frame[cropMarginY:, cropMarginX:width - cropMarginX]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    yellowMask = cv2.inRange(lab, np.array([150, 125, 135]), np.array([255, 200, 170]))
    kernel = np.ones((5, 3), np.uint8)
    combinedMask = cv2.morphologyEx(yellowMask, cv2.MORPH_CLOSE, kernel)
    combinedMask = cv2.morphologyEx(combinedMask, cv2.MORPH_OPEN, kernel)
    maskedRoi = cv2.bitwise_and(roi, roi, mask=combinedMask)
    gray = cv2.cvtColor(maskedRoi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)
    return combinedMask, edges, lines, roi


def classifyLaneLines(lines, cropMarginPercent, height, width, prevLeft=None, prevRight=None, angleThreshold=5):
    leftLines, rightLines = [], []
    leftAngle, rightAngle = None, None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 += int(cropMarginPercent * width)
            x2 += int(cropMarginPercent * width)
            y1 += int(height * 0.3)
            y2 += int(height * 0.3)
            angle = calculateSlopeAngle(x1, y1, x2, y2)
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if 30 < angle < 90:
                if slope < 0:
                    leftLines.extend([(x1, y1), (x2, y2)])
                    leftAngle = angle
                else:
                    rightLines.extend([(x1, y1), (x2, y2)])
                    rightAngle = angle
    if prevLeft and leftAngle is not None:
        if prevLeft['angle'] is not None and abs(leftAngle - prevLeft['angle']) > angleThreshold:
            leftLines = []
    if prevRight and rightAngle is not None:
        if prevRight['angle'] is not None and abs(rightAngle - prevRight['angle']) > angleThreshold:
            rightLines = []
    if not leftLines and prevLeft:
        leftLines = prevLeft['points']
    if not rightLines and prevRight:
        rightLines = prevRight['points']
    currentLeft = {
        'points': leftLines,
        'angle': leftAngle if leftLines else (prevLeft['angle'] if prevLeft else None),
        'slope': (leftLines[1][0] - leftLines[0][0]) / (leftLines[1][1] - leftLines[0][1]) if len(
            leftLines) >= 2 else None
    } if leftLines else None
    currentRight = {
        'points': rightLines,
        'angle': rightAngle if rightLines else (prevRight['angle'] if prevRight else None),
        'slope': (rightLines[1][0] - rightLines[0][0]) / (rightLines[1][1] - rightLines[0][1]) if len(
            rightLines) >= 2 else None
    } if rightLines else None
    return leftLines, rightLines, currentLeft, currentRight


def shadeLaneRegion(frame, leftLines, rightLines):
    height = frame.shape[0]
    overlay = np.zeros_like(frame)
    if leftLines and rightLines and len(leftLines) >= 2 and len(rightLines) >= 2:
        leftPoints = np.array(leftLines).reshape(-1, 2)
        rightPoints = np.array(rightLines).reshape(-1, 2)
        leftFit = np.polyfit(leftPoints[:, 1], leftPoints[:, 0], 1)
        rightFit = np.polyfit(rightPoints[:, 1], rightPoints[:, 0], 1)
        plotY = np.linspace(height // 2, height - 1, num=height // 2)
        leftX = leftFit[0] * plotY + leftFit[1]
        rightX = rightFit[0] * plotY + rightFit[1]
        ptsLeft = np.array([np.transpose(np.vstack([leftX, plotY]))])
        ptsRight = np.array([np.flipud(np.transpose(np.vstack([rightX, plotY])))])
        pts = np.hstack((ptsLeft, ptsRight)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
    return overlay


def identifyObstacles(frame):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerOrange = np.array([0, 45, 50])
    upperOrange = np.array([15, 255, 255])
    lowerBlue = np.array([90, 30, 10])
    upperBlue = np.array([135, 255, 70])
    orangeMask = cv2.inRange(hsvFrame, lowerOrange, upperOrange)
    blueMask = cv2.inRange(hsvFrame, lowerBlue, upperBlue)
    obstacleMask = cv2.bitwise_or(orangeMask, blueMask)
    contours, _ = cv2.findContours(obstacleMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detectedObstacles = []
    minContourArea = 500
    for contour in contours:
        if cv2.contourArea(contour) > minContourArea:
            x, y, w, h = cv2.boundingRect(contour)
            detectedObstacles.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Obstacle", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return detectedObstacles


def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (960, 540)})
    picam2.configure(config)
    picam2.start()

    laneStateMachine = LaneStateMachine(stabilityThreshold=5)
    cropMarginPercent = 0.1
    prevLeft = None
    prevRight = None

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width, _ = frame.shape

        laneMask, edges, lines, laneRoi = detectLanes(frame, cropMarginPercent)
        leftLines, rightLines, currentLeft, currentRight = classifyLaneLines(lines, cropMarginPercent, height, width,
                                                                             prevLeft, prevRight)
        prevLeft = currentLeft if currentLeft else prevLeft
        prevRight = currentRight if currentRight else prevRight
        laneOverlay = shadeLaneRegion(frame, leftLines, rightLines)

        centerLine = None
        if leftLines and rightLines and len(leftLines) >= 2 and len(rightLines) >= 2:
            leftPoints = np.array(leftLines).reshape(-1, 2)
            rightPoints = np.array(rightLines).reshape(-1, 2)
            leftFit = np.polyfit(leftPoints[:, 1], leftPoints[:, 0], 1)
            rightFit = np.polyfit(rightPoints[:, 1], rightPoints[:, 0], 1)
            plotY = np.linspace(height // 2, height - 1, num=height // 2)
            leftX = leftFit[0] * plotY + leftFit[1]
            rightX = rightFit[0] * plotY + rightFit[1]
            centerX = (leftX + rightX) / 2
            centerLine = np.array([np.transpose(np.vstack([centerX, plotY]))]).astype(np.int32)
            cv2.polylines(laneOverlay, [centerLine], False, (0, 255, 0), 2)

        detectedBoxes = identifyObstacles(frame)

        if not leftLines and not rightLines:
            currentDecision = "No Lane Detected"
        else:
            currentDecision = "Go Straight"
            if detectedBoxes and laneOverlay.any() and centerLine is not None:
                laneAreaMask = cv2.cvtColor(laneOverlay, cv2.COLOR_BGR2GRAY)
                _, laneAreaMask = cv2.threshold(laneAreaMask, 1, 255, cv2.THRESH_BINARY)
                for (x, y, w, h) in detectedBoxes:
                    obstacleMask = np.zeros_like(laneAreaMask)
                    cv2.rectangle(obstacleMask, (x, y), (x + w, y + h), 255, -1)
                    overlap = cv2.bitwise_and(laneAreaMask, obstacleMask)
                    if cv2.countNonZero(overlap) > 0:
                        obstacleCenterX = x + w // 2
                        minDist = float('inf')
                        closestCenterX = 0
                        for point in centerLine[0]:
                            centerX, centerY = point
                            dist = abs(obstacleCenterX - centerX)
                            if dist < minDist:
                                minDist = dist
                                closestCenterX = centerX
                        if obstacleCenterX < closestCenterX:
                            currentDecision = "Turn RIGHT - Obstacle on LEFT"
                        else:
                            currentDecision = "Turn LEFT - Obstacle on RIGHT"
                        break

        decision = laneStateMachine.update(currentDecision)
        combinedFrame = cv2.addWeighted(frame, 0.8, laneOverlay, 0.5, 0)

        if "No Lane" in decision:
            textColor = (255, 255, 0)
        elif "STOP" in decision or "Turn" in decision:
            textColor = (0, 0, 255)
        else:
            textColor = (0, 255, 0)

        cv2.putText(combinedFrame, f"Decision: {decision}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)
        for (x, y, w, h) in detectedBoxes:
            cv2.rectangle(combinedFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(combinedFrame, "Obstacle", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Lane and Obstacle Detection", combinedFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
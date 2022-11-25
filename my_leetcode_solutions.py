# 79. Word Search
# Given an m x n grid of characters board and a string word, return true
# if word exists in the grid.
# The word can be constructed from letters of sequentially adjacent cells,
# where adjacent cells are horizontally or vertically neighboring.
# The same letter cell may not be used more than once.

# You are given an integer array height of length n. There are n vertical lines drawn such that
# the two endpoints of the ith line are (i, 0) and (i, height[i]).
#
# Find two lines that together with the x-axis form a container,
# such that the container contains the most water.
#
# Return the maximum amount of water a container can store.
#
# Notice that you may not slant the container.

def maxArea(height) -> int:
    most_water, left, right = 0, 0, len(height) - 1
    while left < right:
        curr_water = min(height[left], height[right]) * (right - left)
        most_water = max(most_water, curr_water)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return most_water
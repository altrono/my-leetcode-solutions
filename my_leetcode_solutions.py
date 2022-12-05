# 1. Two Sum
# Given an array of integers nums and an integer target,
# return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution,
# and you may not use the same element twice.
# You can return the answer in any order.
def two_sum(nums, target):
    my_dict = {}
    for idx in range(len(nums)):
        curr_left = target - nums[idx]
        if curr_left in my_dict.keys():
            return [my_dict[curr_left], idx]
        else:
            my_dict[nums[idx]] = idx
    return []

# nums = [2,7,11,15]
# target = 9
# nums = [3,2,4]
# target = 6
# nums = [3,3]
# target = 6
# print(two_sum(nums, target))



# 42. Trapping Rain Water
# Given n non-negative integers representing an elevation map
# where the width of each bar is 1, compute how much water it can trap after raining.
# T(n) = O(n)
# S(n) = O(1)
def trap(self, height):
    total_water = 0
    left, right = 0, len(height) - 1
    max_left, max_right = 0, 0

    while left < right:
        if height[left] <= height[right]:
            if height[left] >= max_left:
                max_left = height[left]
            else:
                total_water += max_left - height[left]
            left += 1
        else:
            if height[right] >= max_right:
                max_right = height[right]
            else:
                total_water += max_right - height[right]
            right -= 1

    return total_water



# 62. Unique Paths
# There is a robot on an m x n grid. The robot is initially located at the top-left corner
# (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]).
# The robot can only move either down or right at any point in time.
#
# Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

# The test cases are generated so that the answer will be less than or equal to 2 * 109.
def uniquePaths(m, n) :
    if m == n == 1:
        return 1
    dp = [[1] * n] * m
    for row in range(1, len(dp)):
        for col in range(1, len(dp[0])):
            dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
    return dp[-1][-1]
def unique_paths_deepti(m , n):
    if m == n == 1:
        return 1
    dp = [[1] * m] * n

    for row in range(1, len(dp)):
        for col in range(1, len(dp[row])):
            dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
    return dp[-1][-1]


# 63. Unique Paths II
# You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]).
# The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]).
# The robot can only move either down or right at any point in time.
#
# An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot
# include any square that is an obstacle.
#
# Return the number of possible unique paths that the robot can take to reach the bottom-right corner.
#
# The testcases are generated so that the answer will be less than or equal to 2 * 109.

# Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
# Output: 2

def unique_paths_2(obstacleGrid):
    for row in range(len(obstacleGrid)):
        for col in range(len(obstacleGrid[0])):
            if obstacleGrid[row][col] == 1:
                obstacleGrid[row][col] = 0
            elif row == 0 and col == 0:
                obstacleGrid[row][col] = 1
            else:
                if row - 1 >= 0 and col - 1 >= 0:
                    obstacleGrid[row][col] = obstacleGrid[row - 1][col] + obstacleGrid[row][col - 1]
                elif row - 1 >= 0:
                    obstacleGrid[row][col] = obstacleGrid[row - 1][col]
                else:
                    obstacleGrid[row][col] = obstacleGrid[row][col - 1]

    return obstacleGrid[-1][-1]

# obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
# print(unique_paths_2(obstacleGrid))




# 73. Set Matrix Zeroes
# Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
#
# You must do it in place.
# Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
# Output: [[1,0,1],[0,0,0],[1,0,1]]
#
# Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
# Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
def setZeroes(self, matrix):
    """
    Do not return anything, modify matrix in-place instead.
    """
    r_flag = c_flag = False
    rows = len(matrix)
    cols = len(matrix[0])

    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == 0:
                if row == 0:
                    r_flag = True
                if col == 0:
                    c_flag = True
                elif row != 0 and col != 0:
                    matrix[row][0] = 0
                    matrix[0][col] = 0

    for row in range(1, rows):
        for col in range(1, cols):
            if matrix[0][col] == 0 or matrix[row][0] == 0:
                matrix[row][col] = 0

    if r_flag:
        matrix[0] = [0] * cols

    if c_flag:
        for row in range(rows):
            matrix[row][0] = 0





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




# 121. Best Time to Buy and Sell Stock
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
#
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different
# day in the future to sell that stock.
#
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0
# Example 1:
# Input: prices = [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
#
# Example 2:
# Input: prices = [7,6,4,3,1]
# Output: 0
# Explanation: In this case, no transactions are done and the max profit = 0.
def maxProfit(self, prices):
    output = 0
    buy = prices[0]

    for val in prices:
        if val < buy:
            buy = val
        else:
            output = max(output, val - buy)

    return output




# 159. Longest Substring with At Most Two Distinct Characters

# Given a string s, find the length of the longest substring t that contains at most 2 distinct characters.
# input : "eceba"
# output: 3
# input : "ccaabbb"
# output: 5
# T(n) = 0(n)
# S(n) = 0(1)
def longest_substring_with_at_most_two_distinct_characters(s):
    start = 0
    end = 0
    max_len = 0
    d = {}
    while end < len(s):
        d[s[end]] = end
        if len(d) > 2:
            min_ind = min(d.values())
            start = min_ind + 1
            del d[s[min_ind]]
        max_len = max(max_len, end - start + 1)
        end += 1
    return max_len

# s = "ccaabbb"
# s = "abc"
# print(longest_substring_with_at_most_two_distinct_characters(s))





# 221. Maximal Square
# Given an m x n binary matrix filled with 0's and 1's,
# find the largest square containing only 1's and return its area.
# Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# Output: 4
# Input: matrix = [["0","1"],["1","0"]]
# Output: 1


# T(n) = O(mxn)
# S(n) = O(1)
def maximalSquare(matrix):
    output = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if row == 0 or col == 0:
                output = max(output, int(matrix[row][col]))
            elif matrix[row][col] == '1':
                matrix[row][col] = min(int(matrix[row - 1][col]), int(matrix[row][col - 1]),
                                       int(matrix[row - 1][col - 1])) + 1
                output = max(output, int(matrix[row][col]))
    output = output ** 2
    return output

# 238. Product of Array Except Self
# Given an integer array nums, return an array answer such that answer[i] is equal to
# the product of all the elements of nums except nums[i].
#
# The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
#
# You must write an algorithm that runs in O(n) time and without using the division operation.
# Example 1:
#
# Input: nums = [1,2,3,4]
# Output: [24,12,8,6]
# Example 2:
#
# Input: nums = [-1,1,0,-3,3]
# Output: [0,0,9,0,0]
def product_except_self(nums):
    output = [1]
    for i in range(len(nums) - 1, 0, -1):
        output.append(output[-1] * nums[i])
    output = output[::-1]

    left = 1
    for i in range(len(nums)):
        output[i] *= left
        left *= nums[i]
    return output


# 662. Maximum Width of Binary Tree
# Given the root of a binary tree, return the maximum width of the given tree.
#
# The maximum width of a tree is the maximum width among all levels.
#
# The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes),
# where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that
# level are also counted into the length calculation.
#
# It is guaranteed that the answer will in the range of a 32-bit signed integer.
def widthOfBinaryTree(root):
    max_width = 1
    curr_list = [(root, 1)]
    while curr_list:
        next_level = []
        for node, pos in curr_list:
            if node.left:
                next_level.append((node.left, pos * 2))
            if node.right:
                next_level.append((node.right, pos * 2 + 1))

            if next_level != []:
                max_width = max(max_width, next_level[-1][1] - next_level[0][1] + 1)
            curr_list = next_level
    return max_width




# 841. Keys and Rooms
# There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0. Your goal is to visit all the rooms.
# However, you cannot enter a locked room without having its key.
#
# When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, denoting which room it unlocks,
# and you can take all of them with you to unlock the other rooms.
#
# Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited room i,
# return true if you can visit all the rooms, or false otherwise.

def canVisitAllRooms(rooms):
    visited = set()
    stack = [0]

    while stack:
        room = stack.pop()
        visited.add(room)
        for key in rooms[room]:
            if key not in visited:
                stack.append(key)

    return len(visited) == len(rooms)


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


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1, l2):
        output = None
        num_1 = 0
        num_2 = 0

        total = 0

        while l1 is not None:
            num_1 += l1.val
            num_1 *= 10
            l1 = l1.next

        while l2 is not None:
            num_2 += l2.val
            num_2 *= 10
            l2 = l2.next

        total = num_1 + num_2

        while total:
            curr = total % 10
            if output is None:
                output = ListNode()
                output.val = curr
            else:
                output.next = ListNode(curr)
            total = total // 10

        return output
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
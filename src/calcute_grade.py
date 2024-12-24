# Course data from the image (manually extracted)
courses_data = [
    {"name": "硕士生英语", "credits": 32, "grade": 74},
    {"name": "自然辩证法概论", "credits": 18, "grade": 90},
    {"name": "新时代中国特色社会 主义理论与实践", "credits": 36, "grade": 93},
    {"name": "如何写好科研论文", "credits": 1, "grade": 94},
    {"name": "数字信号处理", "credits": 48, "grade": 92},
    {"name": "现代测试技术", "credits": 48, "grade": 83},
    {"name": "机械系统动力学", "credits": 48, "grade": 79},
    {"name": "机电控制技术", "credits": 48, "grade": 94},
    {"name": "数值分析", "credits": 48, "grade": 91},
    {"name": "知识产权法", "credits": 32, "grade": 92},
    {"name": "工程伦理", "credits": 16, "grade": 93},
    {"name": "Java程序设计", "credits": 48, "grade": 95},
    {"name": "国家公务员考试", "credits": 32, "grade": 94},
]

# Calculating the weighted average
total_weighted_grades = sum(course['credits'] * course['grade'] for course in courses_data)
total_credits = sum(course['credits'] for course in courses_data)

weighted_average = total_weighted_grades / total_credits
print(weighted_average)


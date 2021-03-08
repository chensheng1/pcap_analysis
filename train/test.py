list=[[1,2,3,4],[2,3,4,5],[1,2,4,5]]
print(type(list))
result=[]
for i in list:
    max_index = i.index(max(i))
    result.append("概率:" + str(i[max_index]))
print(result)
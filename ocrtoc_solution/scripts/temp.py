origin=['wood_block', 'pudding_box', 'potted_meat_can', 'master_chef_can', 'jenga']
target=['pudding_box', 'wood_block', 'jenga', 'potted_meat_can', 'master_chef_can']

for name in origin:
    print name
    index=target.index(name)
    print target[index]
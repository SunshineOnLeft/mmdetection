
def save_txt(name, pos_gt_bboxes_size_max, gt_bboxes_size_mean, gt_bboxes_size_std, initiallize_csv):
    print("gt_bboxes_size_mean", gt_bboxes_size_mean)
    print("gt_bboxes_size_std", gt_bboxes_size_std)
    if initiallize_csv:
        initiallize_csv = 0
        headers = ['p3','p4','p5','p6','p7']
        with open('assign_gt_distribution/%s_all.txt' %name,'w')as f:
            f.write(",".join(headers) + "\n")
        with open('assign_gt_distribution/%s_mean.txt' %name,'w')as f:
            f.write(",".join(headers) + "\n")
        with open('assign_gt_distribution/%s_std.txt' %name,'w')as f:
            f.write(",".join(headers) + "\n")
    with open('assign_gt_distribution/%s_all.txt' %name,'a')as f:
            f.write("|".join(pos_gt_bboxes_size_max) + "\n")
    with open('assign_gt_distribution/%s_mean.txt' %name,'a')as f:
            f.write(",".join(gt_bboxes_size_mean) + "\n")
    with open('assign_gt_distribution/%s_std.txt' %name,'a')as f:
            f.write(",".join(gt_bboxes_size_std) + "\n")
    print("-----------------------------------------------------")
    with open('assign_gt_distribution/%s_mean.txt' %name, 'r') as f:
        if len(f.readlines()) > 100:
            raise Exception("finish collection")
    return initiallize_csv
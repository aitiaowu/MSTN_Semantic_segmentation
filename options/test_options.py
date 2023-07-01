import argparse

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="test segmentation network")
        parser.add_argument("--model", type=str, default='our_model', help="available option")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")

        parser.add_argument("--data-dir-target", type=str, default='/content/DATA/older_Data/images', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='/content/sourcelist.txt', help="list of images in the target dataset.")

        #parser.add_argument("--data-dir-target", type=str, default='/home/inspectrone/Jay/DATA/older_Data/images', help="Path to the directory containing the source dataset.")
        #parser.add_argument("--data-list-target", type=str, default='/home/inspectrone/Jay/sourcelist.txt', help="Path to the listing of images in the source dataset.")

        parser.add_argument("--num-classes", type=int, default=4, help="Number of classes.")
        parser.add_argument("--set", type=str, default='val', help="choose test set.")
        parser.add_argument("--restore-opt1", type=str, default='/content/drive/MyDrive/Study/Domain_Adaptation_Vessel/MSTN/mstn-master/checkpoint/mstn_module149953', help="restore model parameters from beta1")
        parser.add_argument("--restore-opt2", type=str, default=None, help="restore model parameters from beta2")
        parser.add_argument("--restore-opt3", type=str, default=None, help="restore model parameters from beta3")

        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="restore model parameters from")

        parser.add_argument("--save", type=str, default='/content/drive/MyDrive/Study/Domain_Adaptation_Vessel/MSTN/mstn-master/result', help="Path to save result.")
        #parser.add_argument("--save", type=str, default='/content/DATA/young_Data/labeled_data', help="Path to save result.")
        parser.add_argument('--gt_dir', type=str, default='/content/DATA/older_Data/labeled_data', help='directory for val gt images')
        parser.add_argument('--devkit_dir', type=str, default='/content/sourcelist.txt', help='list directory') 
        #parser.add_argument('--gt_dir', type=str, default='/home/inspectrone/Jay/DATA/older_Data/labeled_data', help='directory for val gt images')
        #parser.add_argument('--devkit_dir', type=str, default='/home/inspectrone/Jay/sourcelist.txt', help='list directory')
        

        return parser.parse_args()
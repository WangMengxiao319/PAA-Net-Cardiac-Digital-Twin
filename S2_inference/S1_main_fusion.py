import argparse
import torch
torch.cuda.empty_cache() # clearing the occupied cuda memory
from torch.backends import cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


from S0_dataset import LoadDataset, NUHDataset_validation
from models.fusion_model import PAA_Net
from utils.loss import calculate_inference_loss, calculate_reconstruction_loss, calculate_fusion_loss,calculate_Dice, evaluate_pointcloud_new, evaluate_AHA_score
from utils.utils import visualize_PC_with_twolabel,visualize_two_PC, ECG_visual_two, visualize_PC_with_twolabel_rotated, lossplot_fusion
                 
import time
import matplotlib
matplotlib.use('Agg')

def train(args):
    args.log_dir = args.log_dir + '/' + args.model_name + '/' + args.ecg_segment
    
    args.log_dir += f'_{args.fusion_method}'
        
    DEVICE = torch.device(f'cuda:{args.GPU_id}') if torch.cuda.is_available() else torch.device('cpu')
    # DEVICE = torch.device('cpu')
    train_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='train')
    val_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    cudnn.benchmark = True
    
    network = PAA_Net(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims, fusion_method=args.fusion_method)

    if args.model is not None:
        print('Loaded trained model from {}.'.format(args.model))
        network.load_state_dict(torch.load(args.model))
    else:
        print('Begin training new model.')

    network.to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)

    max_iter = len(train_dataloader)
    minimum_loss = 1e4
    best_epoch = 0
    
    os.makedirs(args.log_dir, exist_ok=True)

    lossfile_train = args.log_dir + "/training_loss.txt"
    lossfile_val = args.log_dir + "/val_loss.txt"
    lossfile_geometry_train = args.log_dir + "/training_calculate_inference_loss.txt"
    lossfile_geometry_val = args.log_dir + "/val_calculate_inference_loss.txt"
    lossfile_compactness_train = args.log_dir + "/training_compactness_loss.txt"
    lossfile_compactness_val = args.log_dir + "/val_compactness_loss.txt"
    lossfile_RVp_train = args.log_dir + "/training_RVp_loss.txt"
    lossfile_RVp_val = args.log_dir + "/val_RVp_loss.txt"
    lossfile_size_train = args.log_dir + "/training_MIsize_loss.txt"
    lossfile_size_val = args.log_dir + "/val_MIsize_loss.txt"
    
    args_file = args.log_dir + "/args.txt"


    train_time_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        if ((epoch % 25) == 0) and (epoch != 0):  
            plt = lossplot_fusion(lossfile_train, lossfile_val, lossfile_geometry_train, lossfile_geometry_val, lossfile_compactness_train, lossfile_compactness_val, lossfile_RVp_train, lossfile_RVp_val, lossfile_size_train, lossfile_size_val)
            plt.savefig(args.log_dir + '/loss_img.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Initialize the loss accumulators for this epoch.
        train_epoch_losses = {k: 0.0 for k in ['total', 'seg', 'compact', 'kl', 'geo', 'signal', 'size', 'rvp']}

        # if epoch != 0: 
        #     if  lamda_KL < 1:
        #         lamda_KL = 0.1*epoch*args.lamda_KL 
        #     else:
        #         lamda_KL = 0.1

        # training
        network.train()
        total_loss, train_step = 0, 0
        for i, data in enumerate(train_dataloader, 1):
            partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type,filename = data
            partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)      
            partial_input_coarse = partial_input_coarse.to(DEVICE)      
            partial_input = partial_input.permute(0, 2, 1)

            optimizer.zero_grad()

            y_MI = network(partial_input[:, 0:7, :], ECG_input)
       
            loss_seg, loss_compactness, loss_MI_RVpenalty, loss_MI_size = calculate_fusion_loss(y_MI, gt_MI, partial_input)
            loss = loss_seg + args.lamda_compact*loss_compactness + args.lamda_RVp*loss_MI_RVpenalty + args.lamda_MIsize*loss_MI_size 

            check_grad = False
            if check_grad:
                print(loss.requires_grad)
                print(loss_seg.requires_grad)
                print(loss_compactness.requires_grad)
                print(loss_MI_RVpenalty.requires_grad)
                print(loss_MI_size.requires_grad)

            visual_check = False
            if visual_check:
                y_predict = y_MI[0].cpu().detach().numpy()
                y_gd = gt_MI[0].cpu().detach().numpy()
                x_input = partial_input[0].cpu().detach().numpy()
                y_predict_argmax = np.argmax(y_predict, axis=0)
                visualize_PC_with_twolabel(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename='RNmap_gd_pre.jpg')
                

            loss.backward()
            optimizer.step()

            # Accumulate each loss value.
            train_step += 1
            total_loss += loss.item()
            train_epoch_losses['total'] += loss.item()
            train_epoch_losses['seg'] += loss_seg.item()
            train_epoch_losses['compact'] += loss_compactness.item()
            train_epoch_losses['size'] += loss_MI_size.item()
            train_epoch_losses['rvp'] += loss_MI_RVpenalty.item()


            if i % 50 == 0:
                print("Training epoch {}/{}, iteration {}/{}: loss is {}".format(epoch, args.epochs, i, max_iter, loss.item()))
        scheduler.step()
        
        print("\033[96mTraining epoch {}/{}: avg loss = {}\033[0m".format(epoch, args.epochs, total_loss / train_step))
        if epoch % 50 == 0:
            torch.save(network.state_dict(), args.log_dir + '/model_epoch_%d.pkl' % epoch)
        # evaluation
        network.eval()
        val_epoch_losses = {k: 0.0 for k in ['total', 'seg', 'compact', 'kl', 'geo', 'signal', 'size', 'rvp']}
        with torch.no_grad():
            total_loss, val_step = 0, 0
            for i, data in enumerate(val_dataloader, 1):
                partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type,filename = data
                partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)  
                partial_input_coarse = partial_input_coarse.to(DEVICE)  
                partial_input = partial_input.permute(0, 2, 1)

                y_MI = network(partial_input[:, 0:7, :], ECG_input)

                loss_seg, loss_compactness, loss_MI_RVpenalty, loss_MI_size = calculate_fusion_loss(y_MI, gt_MI, partial_input)
                
                loss = loss_seg + args.lamda_compact*loss_compactness + args.lamda_RVp*loss_MI_RVpenalty + args.lamda_MIsize*loss_MI_size 

                total_loss += loss.item()
                val_step += 1

                if ((epoch % 25) == 0) and (epoch != 0) and (i == 1):  
                    y_predict = y_MI[0].cpu().detach().numpy()
                    y_gd = gt_MI[0].cpu().detach().numpy()
                    x_input = partial_input[0].cpu().detach().numpy()
                    y_predict_argmax = np.argmax(y_predict, axis=0)
                    fig = visualize_PC_with_twolabel(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd)
                    fig.savefig(args.log_dir + '/RNmap_gd_pre_val.pdf', dpi=300, bbox_inches='tight')
                    
                val_epoch_losses['total'] += loss.item()
                val_epoch_losses['seg'] += loss_seg.item()
                val_epoch_losses['compact'] += loss_compactness.item()
                val_epoch_losses['size'] += loss_MI_size.item()
                val_epoch_losses['rvp'] += loss_MI_RVpenalty.item()
                
                # Dice = calculate_Dice(y_MI, gt_MI, num_classes=3)
                # print("Validation epoch {}/{}, iteration {}/{}: loss is {}, Dice: {}".format(epoch, args.epochs, i, len(val_dataloader), loss.item(), Dice.cpu().numpy()))
            
            mean_loss = total_loss / val_step
            print("\033[35mValidation epoch {}/{}, loss is {}\033[0m".format(epoch, args.epochs, mean_loss))

            # records the best model and epoch
            if mean_loss < minimum_loss:
                best_epoch = epoch
                minimum_loss = mean_loss           
                strNetSaveName = 'net_model.pkl'
                # strNetSaveName = 'net_with_%d.pkl' % epoch
                torch.save(network.state_dict(), args.log_dir + '/' + strNetSaveName)

        print("\033[4;37mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))
        # Write the training-set averages.
        record_losses([
            lossfile_train, lossfile_geometry_train, lossfile_compactness_train, 
            lossfile_size_train, lossfile_RVp_train
        ], train_epoch_losses, train_step)

        # Write the validation-set averages.
        record_losses([
            lossfile_val, lossfile_geometry_val, lossfile_compactness_val, 
            lossfile_size_val, lossfile_RVp_val
        ], val_epoch_losses, val_step)
        
        print("-----------------------------------------------")
        epoch_end_time = time.time()
        print('epoch train_time: {:.2f} mins'.format((epoch_end_time - epoch_start_time)/60))
    train_time_end = time.time()
    print("Total training time: {:.2f} hours".format((train_time_end - train_time_start)/3600))
    
    with open(args_file, 'a') as f:
        # Write the arguments.
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
        f.write("Total training time: {:.2f} hours\n".format((train_time_end - train_time_start)/3600))
    

    # lossplot(lossfile_train, lossfile_val)

def record_losses(file_list, loss_dict, count):
    # Match each file to its corresponding loss value.
    mapping = zip(file_list, [
        loss_dict['total'], loss_dict['seg'], loss_dict['compact'], 
        loss_dict['size'], loss_dict['rvp']
    ])
    for filepath, total_val in mapping:
        with open(filepath, 'a') as f:
            f.write(f"{total_val / count}\n")
                
def evaluate(args):

    DEVICE = torch.device(f'cuda:{args.GPU_id}') if torch.cuda.is_available() else torch.device('cpu')
    if args.dataset == 'NUS':
        test_dataset = LoadDataset(path=args.partial_root, num_input=args.num_input, split='test')
    elif args.dataset == 'NUS_valid':
        test_dataset = NUHDataset_validation(path=args.partial_root, num_input=args.num_input, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # network = ECGnet(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims)
    if args.model_name == 'PAA_Net':
        network = PAA_Net(in_ch=args.in_ch, out_ch=args.out_ch, num_input=args.num_input, z_dims=args.z_dims, fusion_method=args.fusion_method)
    
    
    args.log_dir = args.log_dir + '/' + args.model_name + '/' + args.ecg_segment
    args.log_dir += f'_{args.fusion_method}'
    model_path = os.path.join(args.log_dir, 'net_model.pkl')
    network.load_state_dict(torch.load(model_path,map_location=DEVICE))
    network.to(DEVICE)

    Dice_Scar, Dice_BZ, Dice_normal,Dice_weighted =  [], [], [], []
    precision_Scar, precision_BZ, precision_normal = [], [], []
    recall_Scar, recall_BZ = [], []
    f1_score_Scar, f1_score_BZ, f1_score_normal = [], [], []
    roc_auc_Scar, roc_auc_BZ, roc_auc_normal = [], [], []
    pre_MI_size_Scar, pre_MI_size_BZ = [], []
    gd_MI_size_Scar, gd_MI_size_BZ = [], []
    MI_center_dist = []
    MI_type_list = []
    AHA_loc_score_list = []
    specificity_normal = []  # euqal to specificity of normal class
    filename_list  = []
    # testing: evaluate the mean loss
    network.eval()
    output_dir = args.evaluate_output_dir + '/' + args.model_name + '/' + args.ecg_segment 
    if args.model_name == 'PAA_Net':
        output_dir += f'_{args.fusion_method}'
    if args.dataset == 'Fudan':
        output_dir  =  output_dir + '/Fudan_test/'
    elif args.dataset == 'NUS_valid':
        output_dir  =  output_dir + '/NUS_valid/'
    output_dir_visual = output_dir + '/visualization/'
    os.makedirs(output_dir_visual, exist_ok=True)
    
    # save each case
    all_PC_list = []
    all_gt_MI_list = []
    all_pred_MI_list = []
    all_ecg_list = []
    
    # For interpretability analysis
    gamma_all = []
    beta_all = []
    def hook_gamma(module, input, output):
        gamma_all.append(output.detach().cpu())

    def hook_beta(module, input, output):
        beta_all.append(output.detach().cpu())
    
    if args.fusion_method == 'film':
        h1 = network.film_layer.fc_gamma.register_forward_hook(hook_gamma)
        h2 = network.film_layer.fc_beta.register_forward_hook(hook_beta)
        
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 1):
            print("Evaluating case {}/{}: ".format(i, len(test_dataloader)))
            partial_input, ECG_input, gt_MI, partial_input_coarse, MI_type, filename = data
            partial_input, ECG_input, gt_MI = partial_input.to(DEVICE), ECG_input.to(DEVICE), gt_MI.to(DEVICE)      
            partial_input_coarse = partial_input_coarse.to(DEVICE)
            partial_input = partial_input.permute(0, 2, 1)

            y_MI = network(partial_input[:, 0:7, :], ECG_input)  # TODO: paritial input, channel 0-2 repesent(x,y,z), 3: represent label
            
            # if args.dataset == 'Fudan':
            #      Dice = calculate_Dice_onlyScar(y_MI, gt_MI, num_classes=2)
            # else:
            Dice = calculate_Dice(y_MI, gt_MI, num_classes=3)
            
            

            # Use the target point count in each class as its weight.
            # if args.dataset == 'Fudan':
            #     num_classes = 2
            # else:
            num_classes = 3
            points_per_class = torch.tensor([(gt_MI == c).sum() for c in range(num_classes)], device=Dice.device, dtype=torch.float)
            if points_per_class.sum() == 0:
                weight = torch.ones_like(points_per_class) / num_classes
            else:
                weight = points_per_class / points_per_class.sum()
            print('weight per classes:', weight)
            weighted_dice = (Dice * weight).sum()
            
            # if args.dataset == 'Fudan':
            #     precision, recall, f1_score, roc_auc,specificity, MI_size_pre, MI_size_gd  = evaluate_pointcloud_new_onlyScar(y_MI, gt_MI, partial_input)
            # elif args.dataset == 'NUS':
            precision, recall, f1_score, roc_auc,specificity, MI_size_pre, MI_size_gd  = evaluate_pointcloud_new(y_MI, gt_MI, partial_input)
        
            Dice_Scar.append(Dice[1].cpu().detach().numpy())
            # mean of each class
            Dice_weighted.append(weighted_dice.cpu().detach().numpy())
            
            precision_Scar.append(precision[1])
            recall_Scar.append(recall[1])
            f1_score_Scar.append(f1_score[1])
            roc_auc_Scar.append(roc_auc[1])

            pre_MI_size_Scar.append(MI_size_pre[1])
            gd_MI_size_Scar.append(MI_size_gd[1])
                
            Dice_normal.append(Dice[0].cpu().detach().numpy())
            precision_normal.append(precision[0])
            roc_auc_normal.append(roc_auc[0])
            f1_score_normal.append(f1_score[0])
            specificity_normal.append(recall[0])   # here we consider normal as positive, so use recall value directly
            
            # if args.dataset == 'NUS':
            Dice_BZ.append(Dice[2].cpu().detach().numpy())
            recall_BZ.append(recall[2])
            precision_BZ.append(precision[2])
            f1_score_BZ.append(f1_score[2])
            roc_auc_BZ.append(roc_auc[2])
            pre_MI_size_BZ.append(MI_size_pre[2])
            gd_MI_size_BZ.append(MI_size_gd[2])
            # elif args.dataset == 'Fudan':
            #     Dice_BZ.append(np.nan)
            #     recall_BZ.append(np.nan)
            #     precision_BZ.append(np.nan)
            #     f1_score_BZ.append(np.nan)
            #     roc_auc_BZ.append(np.nan)
            #     pre_MI_size_BZ.append(np.nan)
            #     gd_MI_size_BZ.append(np.nan)
                


            if MI_type[0] != 'healthy':
                try:
                    center_distance, AHA_loc_score = evaluate_AHA_score(y_MI, gt_MI, partial_input)
                except:
                    # FIXME: Temporary fallback when the prediction contains no scar class (predictions[1] has no 1).
                    center_distance = 0
                    AHA_loc_score = 0

                MI_center_dist.append(center_distance)
                AHA_loc_score_list.append(AHA_loc_score)

            else:
                MI_center_dist.append(np.nan)
                AHA_loc_score_list.append(np.nan)

            MI_type_list.append(MI_type[0]) # batch (1,)
            print(filename)
            filename_list.append(filename[0]) # batch (1,)


            

            gd_ECG = ECG_input[0].cpu().detach().numpy()
            # ECG_visual_two(y_ECG, gd_ECG, filename='ECG_recon.pdf')
            y_predict = y_MI[0].cpu().detach().numpy()
            y_gd = gt_MI[0].cpu().detach().numpy()
            x_input = partial_input[0].cpu().detach().numpy()
            y_predict_argmax = np.argmax(y_predict, axis=0)
            
            all_PC_list.append(x_input.transpose())
            all_gt_MI_list.append(y_gd)
            all_pred_MI_list.append(y_predict_argmax)
            all_ecg_list.append(gd_ECG)
            
            
            if args.visual_check_evaluate:
                visualize_PC_with_twolabel(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename=f'{output_dir_visual}PC_{MI_type[0]}_{filename[0]}.pdf')
                visualize_PC_with_twolabel_rotated(x_input[0:3, 0:args.num_input].transpose(), y_predict_argmax, y_gd, filename=f'{output_dir_visual}PC_{MI_type[0]}_{filename[0]}_rotated.pdf')
                # visualize_two_PC(x_input[0:3, 0:args.num_input].transpose(), y_output[0:3, 0:args.num_input].transpose(), y_gd, filename='PC_recon.pdf')
        
        if args.fusion_method == 'film':
            h1.remove()
            h2.remove()
            gamma_all = torch.cat(gamma_all, dim=0)  # [N, input_dim]
            beta_all  = torch.cat(beta_all, dim=0)
            # save gamma and beta for interpretability analysis
            np.savez_compressed(output_dir + '/film_gamma_beta.npz', gamma=gamma_all.numpy(), beta=beta_all.numpy())

        results_dict = {'filename': filename_list,'MI_type': MI_type_list, 'Dice_Scar': Dice_Scar, 'Dice_BZ': Dice_BZ, 'Dice_normal': Dice_normal, 'Dice_mean': Dice_weighted,
                'precision_normal': precision_normal ,'precision_Scar': precision_Scar, 'precision_BZ': precision_BZ,
                'recall_Scar': recall_Scar, 'recall_BZ': recall_BZ,  'specificity_normal': specificity_normal,
                'f1_score_normal':f1_score_normal, 'f1_score_Scar': f1_score_Scar, 'f1_score_BZ': f1_score_BZ,
                'roc_auc_normal':roc_auc_normal, 'roc_auc_Scar': roc_auc_Scar, 'roc_auc_BZ': roc_auc_BZ,
                'pre_MI_size_Scar': pre_MI_size_Scar, 'pre_MI_size_BZ': pre_MI_size_BZ,
                'gd_MI_size_Scar': gd_MI_size_Scar, 'gd_MI_size_BZ': gd_MI_size_BZ
                , 'MI_center_dist': MI_center_dist, 'AHA_loc_score': AHA_loc_score_list}
        
        df = pd.DataFrame(results_dict)

        df.to_csv(output_dir +'/MI_inference_results.csv', encoding='gbk', index=False)
        
        # save the point cloud results, including input point cloud, ground truth MI label, predicted MI label
        all_PC = np.stack(all_PC_list, axis=0)
        all_gt_MI = np.stack(all_gt_MI_list, axis=0)
        all_pred_MI = np.stack(all_pred_MI_list, axis=0)
        all_filename = np.array(filename_list)
        all_MI_type = np.array(MI_type_list)
        all_ecg = np.stack(all_ecg_list, axis=0)
        np.savez_compressed(output_dir + '/output_PC_MI.npz', PC=all_PC, gt_MI=all_gt_MI, pred_MI=all_pred_MI, filename=all_filename, MI_type=all_MI_type, ecg = all_ecg)
        

        print('Evaluation finished, well done!')       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial_root', type=str, default='./dataset/')
    parser.add_argument('--model', type=str, default=None) #'log/net_model.pkl'
    parser.add_argument('--in_ch', type=int, default=3+4) # coordinate dimension + label index
    parser.add_argument('--out_ch', type=int, default=3) # 3scar, BZ, normal/ 18 for ecg-based classification
    parser.add_argument('--z_dims', type=int, default=16)
    parser.add_argument('--num_input', type=int, default=1024*4)
    parser.add_argument('--batch_size', type=int, default=4) # 4
    parser.add_argument('--lamda_recon', type=float, default=1) # 1
    parser.add_argument('--lamda_KL', type=float, default=1e-2) # 1e-2
    parser.add_argument('--lamda_MIsize', type=float, default=1) # 1
    parser.add_argument('--lamda_RVp', type=float, default=1) # 1 
    parser.add_argument('--lamda_compact', type=float, default=1) # 1
    parser.add_argument('--base_lr', type=float, default=1e-4) #1e-4
    parser.add_argument('--lr_decay_steps', type=int, default=50) 
    parser.add_argument('--lr_decay_rate', type=float, default=0.5) 
    parser.add_argument('--weight_decay', type=float, default=1e-3) #1e-3
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--evaluate_output_dir', type=str, default='output')
    parser.add_argument('--GPU_id', type=int, default=0)
    parser.add_argument('--phase', type=str, default='train') # train / test
    parser.add_argument('--ecg_segment', type=str,default='QRST')  # QRS, ST, QRST
    parser.add_argument('--visual_check_evaluate', action='store_true', help='whether to visualize the point cloud results during evaluation')
    parser.add_argument('--model_name', type=str, default='PAA_Net')   # PAA_Net
    parser.add_argument('--fusion_method', type=str, default='concat')  # concat, cross_attention_2, film
    parser.add_argument('--dataset', type=str, default='NUS_valid')  #  NUS, NUS_valid
    args = parser.parse_args()
    
    if args.phase == 'train':
        train(args)
    else:
        evaluate(args)

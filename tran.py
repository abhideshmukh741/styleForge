import argparse
from pathlib import Path
import torch
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.utils import *
from utils.utils import get_transformer
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='./content_data', help='location of content dataset')
    parser.add_argument('--style_dir', type=str, default='./style_data', help='location of style dataset')
    parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth', help='location of vgg model')
    parser.add_argument('--experiment', type=str, default='experiment1', help='location of experiment')

    parser.add_argument('--final_size', type=int, default=512, help='final size of images')
    parser.add_argument('--content_size', type=int, default=256, help='size of content images')
    parser.add_argument('--style_size', type=int, default=256, help='size of style images')
    parser.add_argument('--crop', action='store_true', help='whether to crop images')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='learning rate decay')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--content_weight',type=float,default=1.0,help='Content weight')
    parser.add_argument('--style_weight',type=float,default=10,help='style weight')
    parser.add_argument('--log_interval', type=int, default=1, help='interval for logging')
    parser.add_argument('--save_interval', type=int, default=2, help='interval for saving model')
    parser.add_argument('--resume', action='store_true', help='whether to resume training')
    parser.add_argument('--decoder_path', type=str,default=None, help='path to decoder model for resuming training')
    parser.add_argument('--optimizer_path', type=str,default=None, help='path to optimizer model for resuming training')

    return parser.parse_args()

def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir=Path('experiment')/args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir/'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    contener_transformer = get_transformer(
        final_size=args.final_size,
        size=args.content_size,
        crop=args.crop
    )

    style_transformer = get_transformer(
        final_size=args.final_size,
        size=args.style_size,
        crop=args.crop
    )

    content_data=imagefolderdataset(args.content_dir,transform=contener_transformer)
    style_data=imagefolderdataset(args.style_dir,transform=style_transformer)

    content_loader=DataLoader(content_data,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    style_loader=DataLoader(style_data,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)


    encoder=VGGEncoder(args.vgg).to(device)
    decoder=vggDecoder().to(device)


    optimizer=optim.Adam(decoder.parameters(),lr=args.lr)

    scheduler= optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0/(1.0 + args.lr_decay+epoch))
    print("traning..")


    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    mse_loss=torch.nn.MSELoss()
    encoder.eval()

    running_loss=None
    running_closs=None
    running_sloss=None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_loader, style_loader), 
                            total=min(len(content_loader), len(style_loader)))



        running_loss=0
        running_closs=0
        running_sloss=0
        
        for content_batch,style_batch in progress_bar:

            content_batch=content_batch.to(device)
            style_batch=style_batch.to(device)


            c_feats= encoder(content_batch)
            s_feats= encoder(style_batch)

            t = adaptive_instent_normalization(c_feats[-1],s_feats[-1])
            g = decoder(t)

            g_feats=encoder(g)

            loss_c = mse_loss(g_feats[-1],t)*args.content_weight

            loss_s=0
            for g_f,s_f in zip(g_feats,s_feats):
                g_mean,g_std=calc_mean_std(g_f)
                s_mean,s_std=calc_mean_std(s_f)
                loss_s+=mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std) 

            loss_s=loss_s*args.style_weight

            loss= loss_s+loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'loss : {loss.item():4f} Content loss : {loss_c.item():4f} Style loss : {loss_s.item():4f}')

            running_loss+=loss.item()
            running_closs+=loss_c.item()
            running_sloss+=loss_s.item()

        scheduler.step()
        running_loss /=len(content_loader)
        running_sloss /=len(content_loader)
        running_closs /=len(content_loader)

        if(epoch+1)%args.log_interval==0:
            tqdm.write(f'Item {epoch+1}: loss {running_loss:4f} Content loss : {running_closs:4f} Style loss : {running_sloss:4f}')





        if (epoch+1)%args.save_interval==0:
            torch.save(decoder.state_dict(),save_dir/f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(),save_dir/f'optimizer_{epoch+1}.pth')
            with torch.no_grad():
                output=torch.cat((content_batch,style_batch,g),dim=0)
                save_image(output,save_dir/f'output_{epoch+1}.jpg',nrow=args.batch_size)





if __name__ == "__main__":
    main()
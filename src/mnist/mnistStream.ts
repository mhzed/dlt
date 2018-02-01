import { Readable } from "stream";
import * as _ from 'lodash'
import { BufQueue } from "bufqueue";

const toInt = (buf: Uint8Array): number => {
  return Buffer.from(buf.buffer, buf.byteOffset, buf.byteLength).readInt32BE(0)
}
// bind this type to the 'data' event
export interface MnistStreamData {
  label: number,
  image: Uint8Array
}
export class MnistStream extends Readable {
  
  constructor(private image: Readable, private label: Readable) {    
    super({objectMode: true});
    this._attach();
  }

  private imageQueue: Uint8Array[] = [];
  private lableQueue: number[] = [];

  // formats are at http://yann.lecun.com/exdb/mnist/
  private _attach() {
    let imagebuf = new BufQueue();
    let labelbuf = new BufQueue();
    let imageMagic, labelMagic;
    let imageCount: number;
    let labelCount: number;
    let imageWidth: number;
    let imageHeight: number;  

    this.image.on('data', (data: Buffer)=>{
      imagebuf.add(data);
      if (_.isNil(imageMagic) && imagebuf.byteLength>=4) {
        imageMagic = imagebuf.consume(4);
      }
      if (_.isNil(imageCount) && imagebuf.byteLength>=4) {
        imageCount = toInt(imagebuf.consume(4));
      }
      if (_.isNil(imageHeight) && imagebuf.byteLength>=4) {
        imageHeight = toInt(imagebuf.consume(4));
      }
      if (_.isNil(imageWidth) && imagebuf.byteLength>=4) {
        imageWidth = toInt(imagebuf.consume(4));
      }
      while (imagebuf.byteLength >= imageWidth * imageHeight) {
        this.imageQueue.push(imagebuf.consume(imageWidth * imageHeight));
      }
      this._flush();
    })
    this.image.on('end', ()=>{
      this.imageQueue.push(null);
      this._flush();
    })
    this.label.on('end', ()=>{
      this.lableQueue.push(null);
      this._flush();
    })
    this.label.on('data', (data: Buffer)=> {
      labelbuf.add(data);
      if (_.isNil(labelMagic) && labelbuf.byteLength>=4)  {
        labelMagic = labelbuf.consume(4);
      }
      if (_.isNil(labelCount) && labelbuf.byteLength>=4) {
        labelCount = toInt(labelbuf.consume(4));
      }
      while(labelbuf.byteLength >= 1) {
        this.lableQueue.push(labelbuf.consume(1)[0]);
      }
      this._flush();
    })
  }

  private _flush(): void {
    while (this.imageQueue.length > 0 && this.lableQueue.length > 0) {
      let obj = {
        image: this.imageQueue.shift(),
        label: this.lableQueue.shift()
      } as MnistStreamData;
      if (obj.image == null) {
        this.push(null);
        return;
      } else {
        let shouldPause = !this.push(obj );
        if (shouldPause) {
          this.image.pause();
          this.label.pause();
          return;
        }
      }
    }
    if (this.image.isPaused()) this.image.resume();
    if (this.label.isPaused()) this.label.resume();
  }
  _read(size) {
    this._flush();
  }


}
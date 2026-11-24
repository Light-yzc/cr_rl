/* Persistent equivalent of Android shell input tap; no game hooks or memory writes. */
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;

public final class CrrlTouch {
    private static Object manager;
    private static Method obtain, inject, recycle, setSource, uptime;
    private static long now() throws Exception { return (Long)uptime.invoke(null); }
    private static void tap(int x, int y) throws Exception {
        long down=now();
        for (int action=0;action<=1;action++) {
            Object event=obtain.invoke(null,down,now(),action,(float)x,(float)y,0);
            try {
                setSource.invoke(event,0x1002);
                if (!Boolean.TRUE.equals(inject.invoke(manager,event,2)))
                    throw new IllegalStateException("Input injection refused");
            } finally { recycle.invoke(event); }
            if (action==0) Thread.sleep(8);
        }
    }
    public static void main(String[] args) throws Exception {
        Class<?> motion=Class.forName("android.view.MotionEvent");
        Class<?> input=Class.forName("android.view.InputEvent");
        Class<?> im=Class.forName("android.hardware.input.InputManager");
        manager=im.getMethod("getInstance").invoke(null);
        inject=im.getMethod("injectInputEvent",input,int.class);
        obtain=motion.getMethod("obtain",long.class,long.class,int.class,float.class,float.class,int.class);
        recycle=motion.getMethod("recycle");
        setSource=motion.getMethod("setSource",int.class);
        uptime=Class.forName("android.os.SystemClock").getMethod("uptimeMillis");
        System.out.println("READY");System.out.flush();
        BufferedReader reader=new BufferedReader(new InputStreamReader(System.in));
        String line;
        while ((line=reader.readLine())!=null) {
            String[] parts=line.trim().split("\\s+");
            if (parts.length==1 && parts[0].equals("PING")) {
                System.out.println("PONG");System.out.flush();continue;
            }
            try {
                if (parts.length!=6 || !parts[0].equals("PLAY")) throw new IllegalArgumentException("Bad command");
                int x0=Integer.parseInt(parts[1]),y0=Integer.parseInt(parts[2]);
                int x1=Integer.parseInt(parts[3]),y1=Integer.parseInt(parts[4]);
                int gap=Integer.parseInt(parts[5]);
                if(x0<0||x0>=900||x1<0||x1>=900||y0<0||y0>=1600||y1<0||y1>=1600||gap<0||gap>200)
                    throw new IllegalArgumentException("Out of calibrated bounds");
                long start=now();tap(x0,y0);Thread.sleep(gap);tap(x1,y1);
                System.out.println("SENT "+(now()-start));
            } catch (Exception error) {
                // A partial sequence must never be retried automatically.
                System.out.println("ERROR "+error.getClass().getSimpleName());
            }
            System.out.flush();
        }
    }
}

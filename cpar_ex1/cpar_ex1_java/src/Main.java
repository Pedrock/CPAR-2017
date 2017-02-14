import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {

    public static void main(String[] args) {

        int i = 1;
        while (i > 0) {
            System.out.println("1. Multiplication");
            System.out.println("2. Line Multiplication");
            int op = readNumber("Selection: ");
            i = readNumber("Dimensions: ");
            switch (op) {
                case 1:
                    OnMult(i, i);
                    break;
                case 2:
                    OnMultLine(i, i);
                    break;
                default:
                    System.out.println("Invalid selection.");
            }
        }
    }

    private static int readNumber(String msg) {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        while (true) {
            System.out.print(msg);
            try{
                return Integer.parseInt(br.readLine());
            }catch(Exception ex){
                System.err.println("Invalid number.");
            }
        }
    }

    static private void initMatrixes(int m_ar, int m_br, double[] pha, double[] phb) {
        int i, j;

        for(i=0; i<m_ar; i++)
            for(j=0; j<m_ar; j++)
                pha[i*m_ar + j] = 1.0;

        for(i=0; i<m_br; i++)
            for(j=0; j<m_br; j++)
                phb[i*m_br + j] = (i+1);
    }

    static private void printResult(int m_ar, int m_br, double[] pha, double[] phb, double[] phc) {
        System.out.print("Result matrix: " );
        for (int i=0; i < 1; i++)
        {	for (int j=0; j < Math.min(10,m_br); j++)
            System.out.print(phc[j] + " ");
        }
        System.out.println();
    }

    static void OnMultLine(int m_ar, int m_br) {
        double[] pha = new double[m_ar * m_br];
        double[] phb = new double[m_ar * m_br];
        double[] phc = new double[m_ar * m_br];
        initMatrixes(m_ar, m_br, pha, phb);

        int i, j, k;
        long startTime = System.nanoTime();

        for (i = 0; i < m_ar; i++)
        {
            for (k = 0; k < m_br; k++)
            {
                for (j = 0; j < m_br; j++)
                {
                    phc[i*m_ar+j] += pha[i*m_ar+k] * phb[k*m_br+j];
                }
            }
        }

        long elapsedTime = System.nanoTime() - startTime;
        double seconds = (double)elapsedTime / 1000000000.0;
        System.out.println("Time: " + seconds + " seconds");
        printResult(m_ar, m_br, pha, phb, phc);
    }

    static void OnMult(int m_ar, int m_br) {
        double[] pha = new double[m_ar * m_br];
        double[] phb = new double[m_ar * m_br];
        double[] phc = new double[m_ar * m_br];
        initMatrixes(m_ar, m_br, pha, phb);

        double temp;
        int i, j, k;
        long startTime = System.nanoTime();

        for(i=0; i < m_ar; i++)
        {
            for(j=0; j < m_br; j++)
            {
                temp = 0;
                for(k=0; k < m_ar; k++)
                {
                    temp += pha[i*m_ar+k] * phb[k*m_br+j];
                }
                phc[i*m_ar+j] = temp;
            }
        }

        long elapsedTime = System.nanoTime() - startTime;
        double seconds = (double)elapsedTime / 1000000000.0;
        System.out.println("Time: " + seconds + " seconds");
        printResult(m_ar, m_br, pha, phb, phc);
    }
}

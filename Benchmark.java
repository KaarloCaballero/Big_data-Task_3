import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;
import jdk.incubator.vector.*;

public class Benchmark {

    private static final int[] MATRIX_SIZES = {64, 128, 256, 512, 1024};
    private static final String[] VECTORIZATION_OPTIONS = {"none", "simd"};
    private static final String[] PARALLELIZATION_OPTIONS = {"none", "threads"};
    private static final int WARMUP_ITERATIONS = 5;
    private static final int REPETITIONS = 15;
    private static final String OUTPUT_CSV = "benchmark_raw_results.csv";
    private static final String MATRIX_DIR = "./matrices";

    public static void main(String[] args) throws InterruptedException, IOException {
        writeCsvHeader(OUTPUT_CSV);

        for (int size : MATRIX_SIZES) {

            // ---------- Load matrices ONCE per size ----------
            DenseMatrix A = loadMatrix("A", size);
            DenseMatrix B = loadMatrix("B", size);

            if (A == null || B == null) {
                System.out.println("[ERROR] Could not load matrices for size " + size);
                continue;
            }

            for (String vec : VECTORIZATION_OPTIONS) {
                for (String par : PARALLELIZATION_OPTIONS) {

                    boolean vectorize = vec.equals("simd");
                    boolean parallelize = par.equals("threads");

                    System.out.printf("[INFO] size=%d, vectorization=%s, parallelization=%s%n",
                            size, vec, par);

                    // ---------- Allocate output matrix once ----------
                    DenseMatrix C = new DenseMatrix(size);

                    for (int iter = 1; iter <= WARMUP_ITERATIONS + REPETITIONS; iter++) {

                        double allocMemMB = getUsedMemoryMB();
                        String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                        String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());
                        boolean warmup = iter <= WARMUP_ITERATIONS;

                        // ---------- Peak memory sampling ----------
                        AtomicLong peakMemoryBytes = new AtomicLong(0);
                        Thread sampler = new Thread(() -> {
                            while (!Thread.currentThread().isInterrupted()) {
                                long used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                                peakMemoryBytes.updateAndGet(prev -> Math.max(prev, used));
                                try { Thread.sleep(1); } catch (InterruptedException e) { break; }
                            }
                        });
                        sampler.setDaemon(true);
                        sampler.start();

                        // ---------- Measure multiplication ----------
                        C.clear(); // reset output matrix before each multiplication
                        long startTime = System.nanoTime();
                        multiplyMatrices(A, B, vectorize, parallelize, C);
                        long endTime = System.nanoTime();

                        sampler.interrupt();
                        sampler.join();

                        double timeMs = (endTime - startTime) / 1e6;
                        double peakMemMB = peakMemoryBytes.get() / (1024.0 * 1024.0);

                        System.out.printf("[%s] size=%d, vec=%s, par=%s | time=%.2f ms | alloc_mem=%.2f MB | peak_mem=%.2f MB | warmup=%b%n",
                                runId, size, vec, par, timeMs, allocMemMB, peakMemMB, warmup);

                        saveCsv(OUTPUT_CSV, Arrays.asList(
                                runId, String.valueOf(size), vec, par,
                                String.format("%.3f", timeMs),
                                String.format("%.3f", allocMemMB),
                                String.format("%.3f", peakMemMB),
                                String.valueOf(iter), timestamp, warmup ? "1" : "0", "No notes"
                        ));

                        Thread.sleep(500); // small pause between iterations
                    }
                }
            }

            // Null matrices to allow GC
            A = null;
            B = null;
        }
    }

    // ------------------- Load dense matrix from binary file ----------
    public static DenseMatrix loadMatrix(String label, int size) throws IOException {
        String filePath = MATRIX_DIR + "/" + label + "_" + size + ".bin";
        if (!Files.exists(Paths.get(filePath))) {
            System.out.println("[ERROR] Matrix file not found: " + filePath);
            return null;
        }

        byte[] bytes = Files.readAllBytes(Paths.get(filePath));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

        DenseMatrix matrix = new DenseMatrix(size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (buffer.remaining() < Integer.BYTES)
                    throw new IOException("Unexpected end of file while reading " + filePath);
                matrix.data[i][j] = (double) buffer.getInt();
            }
        }

        System.out.println("[OK] Loaded matrix '" + label + "_" + size + ".bin' (" + size + "x" + size + ")");
        return matrix;
    }

    // ------------------- Multiply matrices (blocked, SIMD/threads) ----------
    private static void multiplyMatrices(DenseMatrix A, DenseMatrix B,
                                         boolean vectorize, boolean parallelize,
                                         DenseMatrix C) {
        int n = A.size;
        int blockSize = 64;
        DenseMatrix BT = transposeMatrix(B);

        if (parallelize) {
            IntStream.range(0, (n + blockSize - 1) / blockSize).parallel().forEach(iiBlock -> {
                int ii = iiBlock * blockSize;
                for (int jj = 0; jj < n; jj += blockSize) {
                    for (int kk = 0; kk < n; kk += blockSize) {
                        multiplyBlock(A, BT, C, ii, jj, kk, blockSize, vectorize);
                    }
                }
            });
        } else {
            for (int ii = 0; ii < n; ii += blockSize) {
                for (int jj = 0; jj < n; jj += blockSize) {
                    for (int kk = 0; kk < n; kk += blockSize) {
                        multiplyBlock(A, BT, C, ii, jj, kk, blockSize, vectorize);
                    }
                }
            }
        }
    }

    private static void multiplyBlock(DenseMatrix A, DenseMatrix BT, DenseMatrix C,
                                      int ii, int jj, int kk, int blockSize, boolean vectorize) {
        int n = A.size;
        int speciesLength = DoubleVector.SPECIES_PREFERRED.length();

        for (int i = ii; i < Math.min(ii + blockSize, n); i++) {
            for (int j = jj; j < Math.min(jj + blockSize, n); j++) {
                double sum = 0;
                if (vectorize) {
                    int k;
                    for (k = kk; k <= Math.min(kk + blockSize, n) - speciesLength; k += speciesLength) {
                        DoubleVector va = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, A.data[i], k);
                        DoubleVector vb = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, BT.data[j], k);
                        sum += va.mul(vb).reduceLanes(VectorOperators.ADD);
                    }
                    for (; k < Math.min(kk + blockSize, n); k++)
                        sum += A.data[i][k] * BT.data[j][k];
                } else {
                    for (int k = kk; k < Math.min(kk + blockSize, n); k++)
                        sum += A.data[i][k] * BT.data[j][k];
                }
                C.data[i][j] += sum;
            }
        }
    }

    private static DenseMatrix transposeMatrix(DenseMatrix M) {
        int n = M.size;
        DenseMatrix T = new DenseMatrix(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                T.data[j][i] = M.data[i][j];
        return T;
    }

    private static double getUsedMemoryMB() {
        long usedBytes = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        return usedBytes / (1024.0 * 1024.0);
    }

    // ------------------- Dense matrix class ----------
    static class DenseMatrix {
        int size;
        double[][] data;

        DenseMatrix(int n) {
            this.size = n;
            this.data = new double[n][n];
        }

        void clear() {
            for (int i = 0; i < size; i++)
                Arrays.fill(data[i], 0.0);
        }
    }

    // ------------------- CSV helpers ----------
    private static void saveCsv(String path, List<String> row) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(path, true))) {
            writer.println(String.join(";", row));
        } catch (IOException e) {
            System.out.println("[ERROR] CSV append failed: " + e.getMessage());
        }
    }

    private static void writeCsvHeader(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
                writer.println(String.join(";", Arrays.asList(
                        "run_id", "matrix_size", "vectorization", "parallelization",
                        "execution_time_ms", "alloc_mem_mb", "peak_mem_mb",
                        "repetition", "timestamp", "warm-up", "notes"
                )));
            } catch (IOException e) {
                System.out.println("[ERROR] CSV header write failed: " + e.getMessage());
            }
        }
    }
}

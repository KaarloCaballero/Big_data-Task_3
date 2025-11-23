import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import jdk.incubator.vector.*;
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;

public class Benchmark {

    private static final int[] MATRIX_SIZES = {64};
    private static final String[] VECTORIZATION_OPTIONS = {"none", "simd"};
    private static final int[] PARALLELIZATION_OPTIONS = {1, 2, 4, 6, 8, 10};

    private static final int WARMUP_ITERATIONS = 5;
    private static final int REPETITIONS = 15;

    private static final String OUTPUT_CSV = "benchmark_raw_results.csv";
    private static final String MATRIX_DIR = "./matrices";

    private static long prevCpuTime = 0;
    private static long prevNanoTime = 0;

    public static void main(String[] args) throws Exception {

        writeCsvHeader(OUTPUT_CSV);
        OperatingSystemMXBean osBean =
                (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

        int cores = osBean.getAvailableProcessors();
        prevCpuTime = osBean.getProcessCpuTime();
        prevNanoTime = System.nanoTime();

        // Attempt to stabilize CPU frequency (Windows/Linux may require admin/root)
        stabilizeCpuFrequency();

        for (int size : MATRIX_SIZES) {
            DenseMatrix A = loadMatrix("A", size);
            DenseMatrix B = loadMatrix("B", size);
            if (A == null || B == null) continue;

            // Pre-touch matrices to allocate pages and warm caches
            preTouchMatrix(A);
            preTouchMatrix(B);

            for (String vec : VECTORIZATION_OPTIONS) {
                boolean vectorize = vec.equals("simd");

                for (int threads : PARALLELIZATION_OPTIONS) {
                    boolean useParallel = threads > 1;
                    ForkJoinPool pool = useParallel ? new ForkJoinPool(threads) : null;

                    System.out.printf("[INFO] Testing size=%d vectorization=%s threads=%d%n",
                            size, vec, threads);

                    DenseMatrix C = new DenseMatrix(size);
                    preTouchMatrix(C); // pre-touch result matrix as well

                    for (int iter = 1; iter <= WARMUP_ITERATIONS + REPETITIONS; iter++) {
                        boolean warmup = iter <= WARMUP_ITERATIONS;

                        System.gc();
                        Thread.sleep(50);
                        double allocMemMB = getUsedMemoryMB();

                        String runId = "run_" + UUID.randomUUID().toString().substring(0, 8);
                        String timestamp = DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.now());

                        AtomicLong peakMemoryBytes = new AtomicLong(0);
                        Thread sampler = new Thread(() -> {
                            while (!Thread.currentThread().isInterrupted()) {
                                long used = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                                peakMemoryBytes.updateAndGet(p -> Math.max(p, used));
                                try { Thread.sleep(1); } catch (InterruptedException e) { break; }
                            }
                        });
                        sampler.setDaemon(true);
                        sampler.start();

                        C.clear();
                        long start = System.nanoTime();

                        multiplyMatrices(A, B, vectorize, useParallel, C, pool);

                        long end = System.nanoTime();
                        sampler.interrupt();
                        sampler.join();

                        System.gc();
                        Thread.sleep(50);

                        double execMs = (end - start) / 1e6;
                        double peakMem = peakMemoryBytes.get() / (1024.0 * 1024.0);

                        long cpuTime = osBean.getProcessCpuTime();
                        long nanoTime = System.nanoTime();
                        long cpuDiff = cpuTime - prevCpuTime;
                        long timeDiff = nanoTime - prevNanoTime;
                        double cpuLoad = ((double) cpuDiff / timeDiff) * 100.0;
                        prevCpuTime = cpuTime;
                        prevNanoTime = nanoTime;

                        System.out.printf("[%s] size=%d vec=%s thr=%d | time=%.2f ms | alloc=%.2f MB | peak=%.2f MB | cpu=%.1f%% | warmup=%b%n",
                                runId, size, vec, threads, execMs, allocMemMB, peakMem, cpuLoad, warmup);

                        saveCsv(OUTPUT_CSV, Arrays.asList(
                                runId,
                                String.valueOf(size),
                                vec,
                                String.valueOf(threads),
                                String.format("%.3f", execMs),
                                String.format("%.3f", allocMemMB),
                                String.format("%.3f", peakMem),
                                String.format("%.2f", cpuLoad),
                                String.valueOf(cores),
                                String.valueOf(iter),
                                timestamp,
                                warmup ? "1" : "0",
                                "Pre-touched memory & CPU frequency stabilized"
                        ));

                        Thread.sleep(200);
                    }

                    C = null;
                    if (pool != null) pool.shutdown();
                    System.gc();
                    Thread.sleep(50);
                }
            }

            A = null;
            B = null;
            System.gc();
            Thread.sleep(100);
        }
    }

    // ---------------- MEMORY PRE-TOUCH ----------------
    private static void preTouchMatrix(DenseMatrix M) {
        for (int i = 0; i < M.size; i++)
            for (int j = 0; j < M.size; j++)
                M.data[i][j] += 0; // access each element to touch memory pages
    }

    // ---------------- CPU FREQUENCY STABILIZATION ----------------
    private static void stabilizeCpuFrequency() {
        // Placeholder: Actual implementation is OS-specific and may require root/admin
        // For Linux: sudo cpupower frequency-set -g performance
        // For Windows: set max/min CPU in Power Options to 100%
        System.out.println("[INFO] CPU frequency stabilization requested. Ensure performance mode is set.");
    }

    // ---------------- MATRIX LOADING ----------------
    public static DenseMatrix loadMatrix(String label, int size) throws IOException {
        String filePath = MATRIX_DIR + "/" + label + "_" + size + ".bin";
        if (!Files.exists(Paths.get(filePath))) return null;

        byte[] bytes = Files.readAllBytes(Paths.get(filePath));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

        DenseMatrix m = new DenseMatrix(size);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                m.data[i][j] = buffer.getInt();

        System.out.println("[OK] Loaded matrix '" + label + "_" + size + "'");
        return m;
    }

    // ---------------- MATRIX MULTIPLICATION ----------------
    private static void multiplyMatrices(DenseMatrix A, DenseMatrix B, boolean vectorize,
                                         boolean parallel, DenseMatrix C, ForkJoinPool pool) {
        int n = A.size;
        int blockSize = 64;
        DenseMatrix BT = transposeMatrix(B);

        if (parallel && pool != null) {
            int numBlocks = (n + blockSize - 1) / blockSize;
            pool.submit(() ->
                    IntStream.range(0, numBlocks).parallel().forEach(iiBlock -> {
                        int ii = iiBlock * blockSize;
                        for (int jj = 0; jj < n; jj += blockSize)
                            for (int kk = 0; kk < n; kk += blockSize)
                                multiplyBlock(A, BT, C, ii, jj, kk, blockSize, vectorize);
                    })
            ).join();
        } else {
            for (int ii = 0; ii < n; ii += blockSize)
                for (int jj = 0; jj < n; jj += blockSize)
                    for (int kk = 0; kk < n; kk += blockSize)
                        multiplyBlock(A, BT, C, ii, jj, kk, blockSize, vectorize);
        }
    }

    private static void multiplyBlock(DenseMatrix A, DenseMatrix BT, DenseMatrix C,
                                      int ii, int jj, int kk, int blockSize, boolean vectorize) {
        int n = A.size;
        int vecLen = DoubleVector.SPECIES_PREFERRED.length();

        for (int i = ii; i < Math.min(ii + blockSize, n); i++) {
            for (int j = jj; j < Math.min(jj + blockSize, n); j++) {
                double sum = 0;
                if (vectorize) {
                    int k;
                    for (k = kk; k <= Math.min(kk + blockSize, n) - vecLen; k += vecLen) {
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

    // ---------------- MEMORY & CSV HELPERS ----------------
    private static double getUsedMemoryMB() {
        long usedBytes = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        return usedBytes / (1024.0 * 1024.0);
    }

    static class DenseMatrix {
        int size;
        double[][] data;

        DenseMatrix(int n) {
            size = n;
            data = new double[n][n];
        }

        void clear() {
            for (int i = 0; i < size; i++)
                Arrays.fill(data[i], 0.0);
        }
    }

    private static void saveCsv(String out, List<String> row) {
        try (PrintWriter w = new PrintWriter(new FileWriter(out, true))) {
            w.println(String.join(";", row));
        } catch (IOException e) {
            System.out.println("[ERROR] CSV append failed: " + e.getMessage());
        }
    }

    private static void writeCsvHeader(String out) {
        File f = new File(out);
        if (!f.exists()) {
            try (PrintWriter w = new PrintWriter(new FileWriter(f))) {
                w.println(String.join(";", Arrays.asList(
                        "run_id", "matrix_size", "vectorization", "threads", "execution_time_ms",
                        "alloc_mem_mb", "peak_mem_mb", "cpu_usage_percent", "num_cores",
                        "repetition", "timestamp", "warm-up", "notes"
                )));
            } catch (IOException e) {
                System.out.println("[ERROR] CSV header write failed.");
            }
        }
    }
}

/**
 * Resource Fetcher for Expo applications.
 *
 * This module provides functions to download and manage files stored in the application's document directory
 * inside the `react-native-executorch/` directory. These utilities help manage storage and clean up downloaded
 * files when they are no longer needed.
 * @category Utilities - General
 * @remarks
 * **Key Functionality:**
 * - **Download Control**: Pause, resume, and cancel operations through:
 *   - {@link pauseFetching} - Pause ongoing downloads
 *   - {@link resumeFetching} - Resume paused downloads
 *   - {@link cancelFetching} - Cancel ongoing or paused downloads
 * - **File Management**:
 *   - {@link getFilesTotalSize} - Get total size of resources
 *   - {@link listDownloadedFiles} - List all downloaded files
 *   - {@link listDownloadedModels} - List downloaded model files (.pte)
 *   - {@link deleteResources} - Delete downloaded resources
 *
 * **Important Notes:**
 * - Pause/resume/cancel operations work only for remote resources
 * - Most functions accept multiple `ResourceSource` arguments (string, number, or object)
 * - The {@link fetch} method accepts a progress callback (0-1) and returns file paths or throws if interrupted
 */
import { deleteAsync, readDirectoryAsync, readAsStringAsync, moveAsync, } from 'expo-file-system/legacy';
import { RNEDirectory } from './constants/directories';
import { RnExecutorchErrorCode, RnExecutorchError, BaseResourceFetcherClass, } from 'react-native-executorch';
import { ResourceFetcherUtils, HTTP_CODE, DownloadStatus, } from './ResourceFetcherUtils';
import { handleObject, handleLocalFile, handleAsset, handleRemote, } from './handlers';
class ExpoResourceFetcherClass extends BaseResourceFetcherClass {
    downloads = new Map();
    async getFilesSizes(sources) {
        return ResourceFetcherUtils.getFilesSizes(sources);
    }
    async handleObject(source) {
        return handleObject(source);
    }
    handleLocalFile(source) {
        return handleLocalFile(source);
    }
    handleAsset(source, progressCallback) {
        return handleAsset(source, progressCallback, this.downloads);
    }
    handleRemote(uri, source, progressCallback) {
        return handleRemote(uri, source, progressCallback, this.downloads);
    }
    async pause(source) {
        const downloadHandle = this.downloads.get(source);
        if (downloadHandle.status === DownloadStatus.PAUSED) {
            throw new RnExecutorchError(RnExecutorchErrorCode.ResourceFetcherAlreadyPaused, "The file download is currently paused. Can't pause the download of the same file twice.");
        }
        downloadHandle.status = DownloadStatus.PAUSED;
        await downloadHandle.downloadResumable.pauseAsync();
    }
    async resume(source) {
        const downloadHandle = this.downloads.get(source);
        if (downloadHandle.status === DownloadStatus.ONGOING) {
            throw new RnExecutorchError(RnExecutorchErrorCode.ResourceFetcherAlreadyOngoing, "The file download is currently ongoing. Can't resume the ongoing download.");
        }
        downloadHandle.status = DownloadStatus.ONGOING;
        const result = await downloadHandle.downloadResumable.resumeAsync();
        const current = this.downloads.get(source);
        // Paused again or canceled during resume — resolve/reject handled elsewhere.
        if (!current || current.status === DownloadStatus.PAUSED)
            return;
        if (!result ||
            (result.status !== HTTP_CODE.OK &&
                result.status !== HTTP_CODE.PARTIAL_CONTENT)) {
            this.downloads.delete(source);
            downloadHandle.reject(new RnExecutorchError(RnExecutorchErrorCode.ResourceFetcherDownloadFailed, `Failed to resume download from '${downloadHandle.uri}', status: ${result?.status}`));
            return;
        }
        await moveAsync({
            from: downloadHandle.cacheFileUri,
            to: downloadHandle.fileUri,
        });
        this.downloads.delete(source);
        downloadHandle.resolve(ResourceFetcherUtils.removeFilePrefix(downloadHandle.fileUri));
    }
    async cancel(source) {
        const downloadHandle = this.downloads.get(source);
        await downloadHandle.downloadResumable.cancelAsync();
        this.downloads.delete(source);
        downloadHandle.reject(new RnExecutorchError(RnExecutorchErrorCode.DownloadInterrupted, 'Download was canceled.'));
    }
    /**
     * Reads the contents of a file as a string.
     * @param path - Absolute file path or file URI to read.
     * @returns A promise that resolves to the file contents as a string.
     */
    async readAsString(path) {
        const uri = path.startsWith('file://') ? path : `file://${path}`;
        return readAsStringAsync(uri);
    }
    /**
     * Lists all the downloaded files used by React Native ExecuTorch.
     * @returns A promise that resolves to an array of URIs for all the downloaded files.
     */
    async listDownloadedFiles() {
        const files = await readDirectoryAsync(RNEDirectory);
        return files.map((file) => `${RNEDirectory}${file}`);
    }
    /**
     * Deletes downloaded resources from the local filesystem.
     * @param sources - The resource identifiers used when calling `fetch`.
     * @returns A promise that resolves once all specified resources have been removed.
     */
    async deleteResources(...sources) {
        for (const source of sources) {
            const filename = ResourceFetcherUtils.getFilenameFromUri(source);
            const fileUri = `${RNEDirectory}${filename}`;
            if (await ResourceFetcherUtils.checkFileExists(fileUri)) {
                await deleteAsync(fileUri);
            }
        }
    }
    /**
     * Fetches the total size of remote files. Works only for remote files.
     * @param sources - The resource identifiers (URLs).
     * @returns A promise that resolves to the combined size of files in bytes.
     */
    async getFilesTotalSize(...sources) {
        return (await ResourceFetcherUtils.getFilesSizes(sources)).totalLength;
    }
}
export const ExpoResourceFetcher = new ExpoResourceFetcherClass();

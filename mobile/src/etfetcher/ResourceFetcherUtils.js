import { RNEDirectory } from './constants/directories';
import { Logger, ResourceFetcherUtils as CoreUtils, HTTP_CODE, DownloadStatus, SourceType, RnExecutorchError, RnExecutorchErrorCode, } from 'react-native-executorch';
import { Asset } from 'expo-asset';
/**
 * @internal
 */
import { getInfoAsync, makeDirectoryAsync } from 'expo-file-system/legacy';
export { HTTP_CODE, DownloadStatus, SourceType };
/**
 * Utility functions for fetching and managing resources.
 * @category Utilities - General
 */
export var ResourceFetcherUtils;
(function (ResourceFetcherUtils) {
    ResourceFetcherUtils.removeFilePrefix = CoreUtils.removeFilePrefix;
    ResourceFetcherUtils.hashObject = CoreUtils.hashObject;
    ResourceFetcherUtils.calculateDownloadProgress = CoreUtils.calculateDownloadProgress;
    ResourceFetcherUtils.triggerHuggingFaceDownloadCounter = CoreUtils.triggerHuggingFaceDownloadCounter;
    ResourceFetcherUtils.triggerDownloadEvent = CoreUtils.triggerDownloadEvent;
    ResourceFetcherUtils.getFilenameFromUri = CoreUtils.getFilenameFromUri;
    function getType(source) {
        if (typeof source === 'object') {
            return SourceType.OBJECT;
        }
        else if (typeof source === 'number') {
            const uri = Asset.fromModule(source).uri;
            if (uri.startsWith('http')) {
                return SourceType.DEV_MODE_FILE;
            }
            return SourceType.RELEASE_MODE_FILE;
        }
        // typeof source == 'string'
        if (source.startsWith('file://')) {
            return SourceType.LOCAL_FILE;
        }
        return SourceType.REMOTE_FILE;
    }
    ResourceFetcherUtils.getType = getType;
    async function getFilesSizes(sources) {
        const results = [];
        let totalLength = 0;
        for (const source of sources) {
            const type = ResourceFetcherUtils.getType(source);
            let length = 0;
            if (type === SourceType.REMOTE_FILE && typeof source === 'string') {
                try {
                    const response = await fetch(source, { method: 'HEAD' });
                    if (!response.ok) {
                        Logger.warn(`Failed to fetch HEAD for ${source}: ${response.status}`);
                        continue;
                    }
                    const contentLength = response.headers.get('content-length');
                    if (!contentLength) {
                        Logger.warn(`No content-length header for ${source}`);
                    }
                    length = contentLength ? parseInt(contentLength, 10) : 0;
                }
                catch (error) {
                    Logger.warn(`Error fetching HEAD for ${source}:`, error);
                    continue;
                }
            }
            const previousFilesTotalLength = totalLength;
            totalLength += length;
            results.push({ source, type, length, previousFilesTotalLength });
        }
        return { results, totalLength };
    }
    ResourceFetcherUtils.getFilesSizes = getFilesSizes;
    async function createDirectoryIfNoExists() {
        if (!(await checkFileExists(RNEDirectory))) {
            try {
                await makeDirectoryAsync(RNEDirectory, { intermediates: true });
            }
            catch (error) {
                throw new RnExecutorchError(RnExecutorchErrorCode.FileWriteFailed, `Failed to create directory at ${RNEDirectory}`, error);
            }
        }
    }
    ResourceFetcherUtils.createDirectoryIfNoExists = createDirectoryIfNoExists;
    async function checkFileExists(fileUri) {
        const fileInfo = await getInfoAsync(fileUri);
        return fileInfo.exists;
    }
    ResourceFetcherUtils.checkFileExists = checkFileExists;
})(ResourceFetcherUtils || (ResourceFetcherUtils = {}));

/**
 * Config-gated fault injection for the Doppler provider.
 *
 * Injected errors carry `__dopplerFaultInjected = true` so the failure
 * taxonomy marks them `isSimulated: true` in the receipt.
 *
 * @param {{ diagnostics?: { faultInjection?: { enabled?: boolean, failureCode?: string, stage?: string, probability?: number } } }} config
 */
export function createFaultInjector(config: {
    diagnostics?: {
        faultInjection?: {
            enabled?: boolean;
            failureCode?: string;
            stage?: string;
            probability?: number;
        };
    };
}): {
    shouldInject: (currentStage: any) => boolean;
    createInjectedError: () => Error;
};
